import functools

import torch
import torch.nn.functional as F
import tqdm
from einops import rearrange
from torch.utils.data.distributed import DistributedSampler

from .dist_utils import all_gather
import pickle


import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def load_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def compute_vqa(pl_module, batch, test=False):
    infer = pl_module.infer(batch, mask_text=False, mask_image=False)   
    vqa_logits = pl_module.vqa_head(infer["multi_modal_cls_feats"])
    vqa_targets = torch.zeros(len(vqa_logits), pl_module.hparams.config["vqa_label_size"]).to(pl_module.device)

    vqa_labels = batch["vqa_labels"]
    vqa_scores = batch["vqa_scores"]
    
    vqa_answer_types = torch.tensor(batch["answer_types"]).to(pl_module.device)
    ans_tokenizer_dict = load_pickle(pl_module.hparams.config["ans_tokenizer_dict_path"])
    
    ans_tokenizer_feature = torch.stack([ans_tokenizer_dict[each[0]].to(torch.int64) for each in batch["vqa_answer"]], dim=0).to(pl_module.device)
    target_ans_embedding = pl_module.language_encoder.embeddings.word_embeddings(ans_tokenizer_feature)
    target_ans_embedding = torch.mean(target_ans_embedding, dim=1)
    pre_training_knowledge = infer["pre_training_knowledge"]

    for i, (_label, _score) in enumerate(zip(vqa_labels, vqa_scores)):
        for l, s in zip(_label, _score):
            vqa_targets[i, l] = s

    vqa_loss = (F.binary_cross_entropy_with_logits(vqa_logits, vqa_targets) * vqa_targets.shape[1]) + pl_module.eta * (1-F.cosine_similarity(pre_training_knowledge, target_ans_embedding)).sum()

    ret = {
        "vqa_loss": vqa_loss,
        "vqa_logits": vqa_logits,
        "vqa_targets": vqa_targets,
        "vqa_labels": vqa_labels,
        "vqa_scores": vqa_scores,
        "vqa_answer_types": vqa_answer_types,
    }

    if test:
        phase = "test"
    else:
        phase = "train" if pl_module.training else "val"

    loss = getattr(pl_module, f"{phase}_vqa_loss")(ret["vqa_loss"])
    score = getattr(pl_module, f"{phase}_vqa_score")(ret["vqa_logits"], ret["vqa_targets"], ret["vqa_answer_types"])
    pl_module.log(f"vqa/{phase}/loss", loss)
    pl_module.log(f"vqa/{phase}/score", score)

    return ret

@torch.no_grad()
def compute_irtr_recall(pl_module):
    text_dset = pl_module.trainer.datamodule.dms[0].make_no_false_val_dset()
    text_dset.tokenizer = pl_module.trainer.datamodule.dms[0].tokenizer
    text_loader = torch.utils.data.DataLoader(
        text_dset,
        batch_size=256,
        num_workers=pl_module.hparams.config["num_workers"],
        pin_memory=True,
        collate_fn=functools.partial(text_dset.collate,
                                     mlm_collator=pl_module.trainer.datamodule.dms[0].mlm_collator, ), )

    image_dset = pl_module.trainer.datamodule.dms[0].make_no_false_val_dset(image_only=True)
    image_dset.tokenizer = pl_module.trainer.datamodule.dms[0].tokenizer
    dist_sampler = DistributedSampler(image_dset, shuffle=False)
    image_loader = torch.utils.data.DataLoader(
        image_dset,
        batch_size=1,
        num_workers=pl_module.hparams.config["num_workers"],
        sampler=dist_sampler,
        pin_memory=True,
        collate_fn=functools.partial(image_dset.collate,
                                     mlm_collator=pl_module.trainer.datamodule.dms[0].mlm_collator, ), )

    # TODO: speed up the process by caching text/image features
    text_preload = list()
    for _b in tqdm.tqdm(text_loader, desc="text prefetch loop"):
        # == Begin: Add New Keys ==
        batch_text_preload = {
            "text_ids": _b["text_ids"].to(pl_module.device),
            "text_masks": _b["text_masks"].to(pl_module.device),
            "text_labels": _b["text_labels"].to(pl_module.device),
            "img_index": _b["img_index"],
        }
        text_preload.append(batch_text_preload)
        # == End  : Add New Keys ==

    tiids = list()
    for pre in text_preload:
        tiids += pre["img_index"]
    tiids = torch.tensor(tiids)

    image_preload = list()
    for _b in tqdm.tqdm(image_loader, desc="image prefetch loop"):
        image_preload.append((_b['image'][0], _b["img_index"][0]))

    rank_scores = list()
    rank_iids = list()

    for img_batch in tqdm.tqdm(image_preload, desc="rank loop"):
        _im, _iid = img_batch

        img_batch_score = list()
        for txt_batch in text_preload:
            fblen = len(txt_batch["text_ids"])
            im = _im.repeat(fblen, 1, 1, 1).to(device=txt_batch['text_ids'].device)

            with torch.cuda.amp.autocast():
                # == Begin: Add New Keys ==
                batch_infer = {
                    "text_ids": txt_batch["text_ids"],
                    "text_masks": txt_batch["text_masks"],
                    "text_labels": txt_batch["text_labels"],
                }
                score = pl_module.irtr_head(pl_module.infer(batch_infer, img=im, )["multi_modal_cls_feats"])[:, 0]
                # == End  : Add New Keys ==

            img_batch_score.append(score)

        img_batch_score = torch.cat(img_batch_score)
        rank_scores.append(img_batch_score.cpu().tolist())
        rank_iids.append(_iid)

    torch.distributed.barrier()
    gather_rank_scores = all_gather(rank_scores)
    gather_rank_iids = all_gather(rank_iids)

    iids = torch.tensor(gather_rank_iids)
    iids = iids.view(-1)
    scores = torch.tensor(gather_rank_scores)
    scores = scores.view(len(iids), -1)

    topk10 = scores.topk(10, dim=1)
    topk5 = scores.topk(5, dim=1)
    topk1 = scores.topk(1, dim=1)
    topk10_iids = tiids[topk10.indices]
    topk5_iids = tiids[topk5.indices]
    topk1_iids = tiids[topk1.indices]

    tr_r10 = (iids.unsqueeze(1) == topk10_iids).float().max(dim=1)[0].mean()
    tr_r5 = (iids.unsqueeze(1) == topk5_iids).float().max(dim=1)[0].mean()
    tr_r1 = (iids.unsqueeze(1) == topk1_iids).float().max(dim=1)[0].mean()

    topk10 = scores.topk(10, dim=0)
    topk5 = scores.topk(5, dim=0)
    topk1 = scores.topk(1, dim=0)
    topk10_iids = iids[topk10.indices]
    topk5_iids = iids[topk5.indices]
    topk1_iids = iids[topk1.indices]

    ir_r10 = (tiids.unsqueeze(0) == topk10_iids).float().max(dim=0)[0].mean()
    ir_r5 = (tiids.unsqueeze(0) == topk5_iids).float().max(dim=0)[0].mean()
    ir_r1 = (tiids.unsqueeze(0) == topk1_iids).float().max(dim=0)[0].mean()

    return (ir_r1, ir_r5, ir_r10, tr_r1, tr_r5, tr_r10)
