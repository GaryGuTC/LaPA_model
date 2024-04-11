num_gpus=1
per_gpu_batchsize=16

# # === 1. VQA ===
# # === VQA-RAD ===
python main.py with data_root=data/ \
 num_gpus=${num_gpus} \
 num_nodes=1 \
 task_finetune_vqa_vqa_rad \
 per_gpu_batchsize=${per_gpu_batchsize} \
 clip16 \
 text_roberta \
 image_size=384 \
 test_only=True \
 tokenizer=download/roberta-base \
 ans_token_path=download/external_data/VQA_RAD/ans_tokenizer.pt \
 ans_tokenizer_dict_path=download/external_data/VQA_RAD/ans_tokenizer_dict.pkl\
 adj_feat_path=download/external_data/adj_matrix.pt\
 organ_disease_feat=download/external_data/organ_disease_info.pt\
 latent_prompt_size=32\
 load_path=download/checkpoints/VQA_RA_best.ckpt

# # === SLACK ===
python main.py with data_root=data/ \
 num_gpus=${num_gpus} num_nodes=1 \
 task_finetune_vqa_slack \
 per_gpu_batchsize=${per_gpu_batchsize} \
 clip16 text_roberta \
 image_size=384 \
 test_only=True \
 tokenizer=download/roberta-base \
 ans_token_path=download/external_data/SLAKE/ans_tokenizer.pt \
 ans_tokenizer_dict_path=download/external_data/SLAKE/ans_tokenizer_dict.pkl\
 adj_feat_path=download/external_data/adj_matrix.pt\
 organ_disease_feat=download/external_data/organ_disease_info.pt\
 latent_prompt_size=32\
 load_path=download/checkpoints/SLAKE_best.ckpt

# # === MedVQA-2019 ===
python main.py with data_root=data/ \
 num_gpus=${num_gpus} num_nodes=1 \
 task_finetune_vqa_medvqa_2019 \
 per_gpu_batchsize=${per_gpu_batchsize} \
 clip16 text_roberta \
 image_size=384 \
 test_only=True \
 tokenizer=download/roberta-base \
 ans_token_path=download/external_data/VQA_2019/ans_tokenizer.pt \
 ans_tokenizer_dict_path=download/external_data/VQA_2019/ans_tokenizer_dict.pkl\
 adj_feat_path=download/external_data/adj_matrix.pt\
 organ_disease_feat=download/external_data/organ_disease_info.pt\
 latent_prompt_size=32\
 load_path=download/checkpoints/VQA2019_best.ckpt