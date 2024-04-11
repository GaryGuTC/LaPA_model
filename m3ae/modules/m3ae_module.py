import pytorch_lightning as pl
import torch
import torch.nn as nn
from transformers import RobertaConfig, RobertaModel
from transformers.models.bert.modeling_bert import BertConfig, BertModel

from m3ae.modules import objectives, m3ae_utils
from m3ae.modules import prediction_heads
from m3ae.modules.language_encoders.bert_model import BertCrossLayer, LP_fusion_module, pre_training_module, GAT_module
from m3ae.modules.m3ae_utils import init_weights
from m3ae.modules.vision_encoders import swin_transformer as swin
from m3ae.modules.vision_encoders.clip_model import build_model, adapt_position_encoding
from m3ae.modules.vision_encoders.swin_helpers import swin_adapt_position_encoding
import torch
from torch import nn
import copy
cp = copy.deepcopy
from torch.nn import functional as F

class M3AETransformerSS(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        
        # == Begin: Build Models ==
        self.is_clip = ('swin' not in config['vit'])
        if 'roberta' in config['tokenizer']:
            bert_config = RobertaConfig(
                vocab_size=config["vocab_size"],
                hidden_size=config["hidden_size"],
                num_hidden_layers=config["num_layers"],
                num_attention_heads=config["num_heads"],
                intermediate_size=config["hidden_size"] * config["mlp_ratio"],
                max_position_embeddings=config["max_text_len"],
                hidden_dropout_prob=config["drop_rate"],
                attention_probs_dropout_prob=config["drop_rate"],
            )
        elif 'bert' in config['tokenizer']:
            bert_config = BertConfig(
                vocab_size=config["vocab_size"],
                hidden_size=config["hidden_size"],
                num_hidden_layers=config["num_layers"],
                num_attention_heads=config["num_heads"],
                intermediate_size=config["hidden_size"] * config["mlp_ratio"],
                max_position_embeddings=config["max_text_len"],
                hidden_dropout_prob=config["drop_rate"],
                attention_probs_dropout_prob=config["drop_rate"],
            )
        else:
            raise ValueError

        resolution_after = config['image_size']
        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:
                if self.is_clip:
                    build_model(config['vit'], resolution_after=resolution_after)
                else:
                    getattr(swin, self.hparams.config["vit"])(pretrained=True, config=self.hparams.config)
                if 'roberta' in config['tokenizer']:
                    RobertaModel.from_pretrained(config['tokenizer'])
                else:
                    BertModel.from_pretrained(config['tokenizer'])
            torch.distributed.barrier()
        if self.is_clip:
            self.vision_encoder = build_model(config['vit'], resolution_after=resolution_after)
        else:
            self.vision_encoder = getattr(swin, self.hparams.config["vit"])(pretrained=True, config=self.hparams.config)
            self.vision_pooler = nn.AdaptiveAvgPool1d(1)
        if 'roberta' in config['tokenizer']:
            self.language_encoder = RobertaModel.from_pretrained(config['tokenizer'])
        else:
            self.language_encoder = BertModel.from_pretrained(config['tokenizer'])

        self.multi_modal_language_proj = nn.Linear(config['input_text_embed_size'], config['hidden_size'])
        self.multi_modal_language_proj.apply(init_weights)
        self.multi_modal_vision_proj = nn.Linear(config['input_image_embed_size'], config['hidden_size'])
        self.multi_modal_vision_proj.apply(init_weights)

        self.modality_type_embeddings = nn.Embedding(2, config["hidden_size"])
        self.modality_type_embeddings.apply(init_weights)

        
        self.multi_modal_vision_layers = nn.ModuleList(
            [BertCrossLayer(bert_config) for _ in range(config['num_top_layer'])]) # 6
        self.multi_modal_vision_layers.apply(init_weights)
        self.multi_modal_language_layers = nn.ModuleList(
            [BertCrossLayer(bert_config) for _ in range(config['num_top_layer'])])
        self.multi_modal_language_layers.apply(init_weights)
        
        ############################################################
        ################# Latent prompt pipeline for diagnosis #####
        ############################################################
        used_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.latent_prompt_size = config['latent_prompt_size'] #32
        self.organ_disease_feat_size = 577 # it's fixed
        
        self.GAT_feat = self.latent_prompt_size//8 if self.latent_prompt_size >= 8 else 1
        self.alpha = torch.tensor(1).to(used_device)
        self.beta = torch.tensor(0.1).to(used_device)
        self.theta = torch.tensor(0.1).to(used_device)
        self.eta = torch.tensor(0.1).to(used_device)
        
        ################# latent prompt generation
        self.LP_knowledge = nn.Parameter(torch.FloatTensor(self.latent_prompt_size, config["hidden_size"]), requires_grad=True).to(used_device)
        nn.init.normal_(self.LP_knowledge, 0, 1/config["hidden_size"])
        
        ################# 
        self.ans_tokenzer = torch.load(config["ans_token_path"]).to(used_device)
        self.pre_training_MLP = nn.Linear(self.ans_tokenzer.size(1), 128)
        self.pre_training_MLP.apply(init_weights)
        self.pre_training_layer = pre_training_module(config["hidden_size"])
        self.pre_training_layer.apply(init_weights)
        
        ################# LP Fusion
        self.LP_layers = nn.ModuleList(
            [LP_fusion_module(config["hidden_size"]) for _ in range(config['num_top_layer'])])
        self.LP_layers.apply(init_weights)
        
        ################# PK Fusion
        self.adj_feat = torch.load(config["adj_feat_path"]).to(used_device)
        self.organ_disease_feat = torch.load(config["organ_disease_feat"]).to(used_device)
        self.GAT_layer = GAT_module(config["hidden_size"])
        self.organ_average = nn.Conv1d(self.organ_disease_feat_size, self.GAT_feat, 1)
        self.organ_average.apply(init_weights)
        
        ################# Final conclusion
        self.output_average = nn.Conv1d(self.latent_prompt_size + self.GAT_feat,1,1)
        self.output_average.apply(init_weights)
        ############################################################
        ################# Latent prompt pipeline for diagnosis #####
        ############################################################
        
        self.multi_modal_vision_pooler = prediction_heads.Pooler(config["hidden_size"])
        self.multi_modal_vision_pooler.apply(init_weights)
        self.multi_modal_language_pooler = prediction_heads.Pooler(config["hidden_size"])
        self.multi_modal_language_pooler.apply(init_weights)
        # == End  : Build Models ==

        # == Begin: Load Models ==
        if self.hparams.config["load_path"] != "" and not self.hparams.config["test_only"]:
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            if self.is_clip:
                state_dict = adapt_position_encoding(state_dict,
                                                     after=resolution_after,
                                                     patch_size=self.hparams.config['patch_size'])
            else:
                state_dict = swin_adapt_position_encoding(state_dict, after=resolution_after)
            self.load_state_dict(state_dict, strict=False)
        # == End  :  Load Models == 
                       
        # == Begin: Build Heads For Downstream Tasks ==
        hs = self.hparams.config["hidden_size"]
        if self.hparams.config["loss_names"]["vqa"] > 0:
            vs = self.hparams.config["vqa_label_size"] 
            self.vqa_head = nn.Sequential(
                nn.Linear(hs, hs*2),
                nn.LayerNorm(hs*2),
                nn.GELU(),
                nn.Linear(hs*2, vs),
            )
            self.vqa_head.apply(init_weights)

        m3ae_utils.set_metrics(self)
        self.current_tasks = list()
        # == End:  Build Heads For Downstream Tasks ==

        # == Begin: Load Models For Testing ==
        if self.hparams.config["load_path"] != "" and self.hparams.config["test_only"]:
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            state_dict = adapt_position_encoding(state_dict, after=resolution_after,
                                                 patch_size=self.hparams.config['patch_size'])
            self.load_state_dict(state_dict, strict=False)
        # == End:  Load Models For Testing ==

    def infer(
            self,
            batch,
            mask_text=False,
            mask_image=False,
            image_token_type_idx=1,
            img=None,
            output_attentions=False,
            unimodal=False
    ):
        ret = dict()

        # == Begin: Fetch the inputs ==
        if img is None:
            if f"image_{image_token_type_idx - 1}" in batch:
                img_key = f"image_{image_token_type_idx - 1}"
            else:
                img_key = "image"
            img = batch[img_key][0]
        do_mlm = "_mlm" if mask_text else ""
        text_ids = batch[f"text_ids{do_mlm}"]
        text_labels = batch[f"text_labels{do_mlm}"]
        text_masks = batch[f"text_masks"]
        device = text_ids.device
        # == End  : Fetch the inputs ==

        # == Begin: Text Encoding ==
        uni_modal_text_feats = self.language_encoder.embeddings(input_ids=text_ids)
        text_input_shape = text_masks.size()
        extended_text_masks = self.language_encoder.get_extended_attention_mask(text_masks, text_input_shape, device)
        for layer in self.language_encoder.encoder.layer:
            uni_modal_text_feats = layer(uni_modal_text_feats, extended_text_masks)[0]
        uni_modal_text_feats = self.multi_modal_language_proj(uni_modal_text_feats)
        # == End  : Text Encoding ==

        # == Begin: Image Encoding ==
        uni_modal_image_feats = self.vision_encoder(img)
        uni_modal_image_feats = self.multi_modal_vision_proj(uni_modal_image_feats)
        image_masks = torch.ones((uni_modal_image_feats.size(0), uni_modal_image_feats.size(1)), dtype=torch.long,
                                device=device)
        extended_image_masks = self.language_encoder.get_extended_attention_mask(image_masks, image_masks.size(),
                                                                                device)
        # == End  : Image Encoding ==

        # == Begin: Assign Type Embeddings ==
            
        uni_modal_text_feats, uni_modal_image_feats = (
            uni_modal_text_feats + self.modality_type_embeddings(torch.zeros_like(text_masks)),
            uni_modal_image_feats + self.modality_type_embeddings(torch.full_like(image_masks, image_token_type_idx)),
        )
        # == End  : Assign Type Embeddings ==
        
        # == Begin: Multi-Modal Fusion ==
        ret["attentions"] = {"text2image_attns": [], "image2text_attns": []} if output_attentions else None
        
        x, y = uni_modal_text_feats, uni_modal_image_feats 
        LP_detected = self.LP_knowledge.unsqueeze(0).expand(x.size(0), self.LP_knowledge.size(0), self.LP_knowledge.size(1))
        ans_tokenzer = self.language_encoder.embeddings.word_embeddings(self.ans_tokenzer)
        ans_tokenzer_feature = ans_tokenzer.expand(x.size(0), ans_tokenzer.size(1), ans_tokenzer.size(2))
        ans_tokenzer_feature = ans_tokenzer_feature.transpose(1,2) 
        ans_tokenzer_feature = self.pre_training_MLP(ans_tokenzer_feature).transpose(1,2)
        LP_detected = self.pre_training_layer(LP_detected, ans_tokenzer_feature)
        
        pre_training_knowledge = torch.mean(LP_detected, dim=1)
        
        for layer_idx, (text_layer, image_layer, LP_layer) in enumerate(zip(self.multi_modal_language_layers,
                                                                  self.multi_modal_vision_layers,
                                                                  self.LP_layers)):
            # == Begin: Fetch the intermediate outputs (different layers to perform MIM) ==
            if mask_image and self.hparams.config["mim_layer"] == layer_idx:
                ret[f"multi_modal_text_feats_{layer_idx}"], ret[f"multi_modal_image_feats_{layer_idx}"] = x, y
            # == End  : Fetch the intermediate outputs (different layers to perform MIM) ==
            mid_text_feat, x1 = text_layer(x, y, extended_text_masks, extended_image_masks, output_attentions=True)
            mid_img_feat, y1 = image_layer(y, x, extended_image_masks, extended_text_masks, output_attentions=True)
            x, y = x1[0], y1[0] 
            # == End: Co-Attention & prefix_tuning ==
            # == Begin: For visualization: Return the attention weights ==
            if output_attentions:
                ret["attentions"]["text2image_attns"].append(x1[1:])
                ret["attentions"]["image2text_attns"].append(y1[1:])
            # == End  : For visualization: Return the attention weights ==
            tmp_x_y = torch.concat([x, y], dim=1)
            LP_detected = LP_layer(LP_detected, mid_text_feat, mid_img_feat, tmp_x_y)
        
        ############## Final diagnosis
        adj_feat = self.adj_feat.unsqueeze(0).expand(x.size(0), self.adj_feat.size(0), self.adj_feat.size(1))
        organ_disease_feat = self.organ_disease_feat.expand(x.size(0), self.organ_disease_feat.size(1))
        organ_disease_feat = self.language_encoder.embeddings.word_embeddings(organ_disease_feat)
        GAT_feat = self.GAT_layer(LP_detected, organ_disease_feat, adj_feat)
        GAT_feat = self.organ_average(GAT_feat)
        LP_detected = torch.concatenate([LP_detected, GAT_feat], dim=1) # 
        
        # == Begin: == Output Multi-Modal Features ==
        multi_modal_text_feats, multi_modal_image_feats = x, y
        LP_detected = torch.squeeze(self.output_average(LP_detected), dim=1)
        LP_detected = nn.Tanh()(LP_detected)
        
        multi_modal_text_cls_feats = self.multi_modal_language_pooler(x)
        if self.is_clip:
            multi_modal_image_cls_feats = self.multi_modal_vision_pooler(y)
        else:
            avg_image_feats = self.vision_pooler(multi_modal_image_feats.transpose(1, 2)).view(
                multi_modal_image_feats.size(0), 1, -1)
            multi_modal_image_cls_feats = self.multi_modal_vision_pooler(avg_image_feats)
            
        multi_modal_cls_feats = self.beta * multi_modal_text_cls_feats + self.theta * multi_modal_image_cls_feats + self.alpha * LP_detected

        ret.update({
            "images": img,
            "pre_training_knowledge": pre_training_knowledge,
            "text_labels": text_labels,
            "text_ids": text_ids,
            "text_masks": text_masks,
            "extended_image_masks": extended_image_masks,
            "extended_text_masks": extended_text_masks,
            "multi_modal_text_feats": multi_modal_text_feats,
            "multi_modal_image_feats": multi_modal_image_feats,
            "multi_modal_cls_feats": multi_modal_cls_feats,
        })

        return ret

    def forward(self, batch, test=False):
        ret = dict()

        if len(self.current_tasks) == 0:
            ret.update(self.infer(batch))
            return ret

        # Fine-Tuning: Visual Question Answering
        if "vqa" in self.current_tasks:
            ret.update(objectives.compute_vqa(self, batch, test=test))

        return ret

    def training_step(self, batch, batch_idx):
        m3ae_utils.set_task(self)
        output = self(batch)
        total_loss = sum([v * self.hparams.config["loss_names"][k.replace("_loss", "")]
                          for k, v in output.items() if "loss" in k]) # vqa_loss

        return total_loss

    def training_epoch_end(self, outs):
        m3ae_utils.epoch_wrapup(self)

    def validation_step(self, batch, batch_idx):
        m3ae_utils.set_task(self)
        output = self(batch)

    def validation_epoch_end(self, outs):
        m3ae_utils.epoch_wrapup(self)

    def test_step(self, batch, batch_idx):
        m3ae_utils.set_task(self)
        output = self(batch, test=True)

    def test_epoch_end(self, outs):
        m3ae_utils.epoch_wrapup(self, test=True)

    def configure_optimizers(self):
        return m3ae_utils.set_schedule(self)
