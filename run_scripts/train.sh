num_gpus=1
per_gpu_batchsize=16

# === VQA ===
# vqa-rad                     
python main.py with data_root=data/ \
 num_gpus=${num_gpus} \
 num_nodes=1 \
 task_finetune_vqa_vqa_rad \
 per_gpu_batchsize=${per_gpu_batchsize} \
 clip16 \
 text_roberta \
 image_size=384 \
 tokenizer=download/roberta-base \
 load_path=download/pretrained/m3ae.ckpt \
 ans_token_path=download/external_data/VQA_RAD/ans_tokenizer.pt \
 ans_tokenizer_dict_path=download/external_data/VQA_RAD/ans_tokenizer_dict.pkl\
 adj_feat_path=download/external_data/adj_matrix.pt\
 organ_disease_feat=download/external_data/organ_disease_info.pt\
 latent_prompt_size=32\
 max_epoch=20
 
# # vqa-slack
python main.py with data_root=data/ \
 num_gpus=${num_gpus} \
 num_nodes=1 \
 task_finetune_vqa_slack \
 per_gpu_batchsize=${per_gpu_batchsize} \
 clip16 \
 text_roberta \
 image_size=384 \
 tokenizer=download/roberta-base \
 load_path=download/pretrained/m3ae.ckpt \
 ans_token_path=download/external_data/SLAKE/ans_tokenizer.pt \
 ans_tokenizer_dict_path=download/external_data/SLAKE/ans_tokenizer_dict.pkl\
 adj_feat_path=download/external_data/adj_matrix.pt\
 organ_disease_feat=download/external_data/organ_disease_info.pt\
 latent_prompt_size=32\
 clip_resizedcrop
 
# # med-vqa-2019
python main.py with data_root=data/ \
 num_gpus=${num_gpus} \
 num_nodes=1 \
 task_finetune_vqa_medvqa_2019 \
 per_gpu_batchsize=${per_gpu_batchsize} \
 clip16 \
 text_roberta \
 image_size=384 \
 tokenizer=download/roberta-base \
 load_path=download/pretrained/m3ae.ckpt \
 ans_token_path=download/external_data/VQA_2019/ans_tokenizer.pt \
 ans_tokenizer_dict_path=download/external_data/VQA_2019/ans_tokenizer_dict.pkl\
 adj_feat_path=download/external_data/adj_matrix.pt\
 organ_disease_feat=download/external_data/organ_disease_info.pt\
 latent_prompt_size=32\
 clip_resizedcrop