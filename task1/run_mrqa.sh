# Pretraining on MRQA dataset
# python -m torch.distributed.launch --nproc_per_node=4 --master_port=10001 run_qa.py \
python run_qa.py \
 --dataset_name  '../utils/mrqa.py' \
 --dataset_config_name mrqa_rc \
 --model_name_or_path roberta-large \
 --do_train \
 --do_eval \
 --early_stop \
 --early_stopping_patience 3 \
 --version_2_with_negative \
 --logging_steps 500 \
 --save_steps 500 \
 --learning_rate 3e-5  \
 --num_train_epochs 10 \
 --max_seq_length 512  \
 --max_answer_length 50 \
 --doc_stride 128  \
 --cache_dir cache\
 --output_dir save/roberta-large-mrqa \
 --overwrite_output_dir  \
 --per_device_train_batch_size 2 \
 --per_device_eval_batch_size 2 \
 --gradient_accumulation_steps 30  \
 --evaluation_strategy steps \
 --eval_steps  500 \
 --load_best_model_at_end \
 --metric_for_best_model f1 \
 --warmup_steps 1000 \
 --weight_decay 0.01 \
 --fp16 \
#  --sharded_ddp 

# # Pretraining on MRQA small dataset
# python -m torch.distributed.launch --nproc_per_node=4 --master_port=10001 run_qa.py \
#  --dataset_name  '../utils/mrqa.py' \
#  --dataset_config_name mrqa_rc_small \
#  --model_name_or_path roberta-large \
#  --do_train \
#  --do_eval \
#  --early_stop \
#  --early_stopping_patience 3 \
#  --version_2_with_negative \
#  --logging_steps 500 \
#  --save_steps 500 \
#  --learning_rate 3e-5  \
#  --num_train_epochs 10 \
#  --max_seq_length 512  \
#  --max_answer_length 50 \
#  --doc_stride 128  \
#  --cache_dir cache\
#  --output_dir save/roberta-large-mrqa-small \
#  --overwrite_output_dir  \
#  --per_device_train_batch_size 2 \
#  --per_device_eval_batch_size 2 \
#  --gradient_accumulation_steps 30  \
#  --evaluation_strategy epoch \
#  --load_best_model_at_end \
#  --metric_for_best_model f1 \
#  --warmup_steps 1000 \
#  --weight_decay 0.01 \
#  --fp16 \
#  --sharded_ddp 