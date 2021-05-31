# CUDA_VISIBLE_DEVICES=1,3 python -m torch.distributed.launch --nproc_per_node=2 --master_port=10005 legacy.py \
CUDA_VISIBLE_DEVICES=0,2 python legacy.py \
 --dataset_name  '../utils/dialdoc/dialdoc.py'\
 --dataset_config_name doc2dial_rc \
 --model_type microsoft/deberta-large\
 --model_name_or_path microsoft/deberta-large\
 --do_train \
 --do_eval \
 --version_2_with_negative \
 --logging_steps 500 \
 --save_steps 500 \
 --learning_rate 3e-5  \
 --num_train_epochs 5 \
 --max_seq_length 512  \
 --max_answer_length 50 \
 --doc_stride 128  \
 --cache_dir cache\
 --output_dir save/deberta-large \
 --overwrite_output_dir  \
 --per_gpu_train_batch_size 1 \
 --per_gpu_eval_batch_size 1 \
 --gradient_accumulation_steps 30\
 --warmup_steps 1000 \
 --weight_decay 0.01 \
 --fp16 \

# # for inference:
# CUDA_VISIBLE_DEVICES=4 python legacy.py \
#  --dataset_name  '../utils/dialdoc/dialdoc.py'\
#  --dataset_config_name doc2dial_rc\
#  --predict_file '../dataset/doc2dial/v1.0.1/doc2dial_dial_validation.json' \
#  --model_name_or_path 'save/deberta-large/checkpoint-7500' \
#  --model_type microsoft/deberta-large \
#  --do_eval \
#  --version_2_with_negative \
#  --max_seq_length 512  \
#  --max_answer_length 50 \
#  --doc_stride 128  \
#  --output_dir save/deberta-large/dev\
#  --per_gpu_eval_batch_size 8 \
#  --fp16 