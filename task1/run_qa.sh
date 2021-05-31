python -m torch.distributed.launch --nproc_per_node=3 --master_port=10003 run_qa.py \
 --dataset_name  '../utils/dialdoc/dialdoc.py'\
 --dataset_config_name doc2dial_rc \
 --model_name_or_path save/roberta-large-mrqa-continue-seed425/checkpoint-4020 \
 --do_train \
 --do_eval \
 --early_stop \
 --early_stopping_patience 3 \
 --logging_steps 500 \
 --save_steps 500 \
 --learning_rate 3e-5  \
 --num_train_epochs 13 \
 --max_seq_length 512  \
 --max_answer_length 50 \
 --doc_stride 128  \
 --cache_dir cache\
 --output_dir save/roberta-large-mrqa-continue-seed425 \
 --overwrite_output_dir  \
 --per_device_train_batch_size 2 \
 --per_device_eval_batch_size 2 \
 --gradient_accumulation_steps 30  \
 --evaluation_strategy epoch \
 --load_best_model_at_end \
 --early_stopping_patience 3 \
 --metric_for_best_model exact \
 --warmup_steps 1000 \
 --weight_decay 0.01 \
 --save_total_limit 5 \
 --fp16 \
 --seed 425 \
 --sharded_ddp


# # for inference:
# CUDA_VISIBLE_DEVICES=1 python run_qa.py \
#  --dataset_name  '../utils/dialdoc/dialdoc.py'\
#  --dataset_config_name doc2dial_rc_testdev\
#  --validation_file '../dataset/doc2dial/v1.0.1/doc2dial_dial_testdev.json' \
#  --model_name_or_path 'save/roberta-large-coqa-single' \
#  --do_eval \
#  --version_2_with_negative \
#  --max_seq_length 512  \
#  --max_answer_length 50 \
#  --doc_stride 128  \
#  --output_dir save/roberta-large-coqa-single/testdev \
#  --per_device_eval_batch_size 8 \
#  --fp16 
