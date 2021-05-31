python -m torch.distributed.launch --nproc_per_node=2 --master_port=10001 run_qa.py \
 --dataset_name '../utils/dialdoc/dialdoc.py' \
 --dataset_config_name doc2dial_rc_testdev \
 --model_name_or_path /home/etsuko/dialdoc/task1/save/roberta-large \
 --do_eval \
 --logging_steps 2000 \
 --save_steps 2000 \
 --num_train_epochs 0 \
 --max_seq_length 512  \
 --max_answer_length 50 \
 --doc_stride 128  \
 --cache_dir cache \
 --output_dir save/roberta-large/testdev \
 --overwrite_output_dir  \
 --per_device_eval_batch_size 2  \
 --gradient_accumulation_steps 15  \
 --fp16 \
 --sharded_ddp

# CUDA_VISIBLE_DEVICES=4 python run_qa.py \
#  --dataset_name  '../utils/dialdoc/dialdoc.py'\
#  --dataset_config_name doc2dial_rc_testdev\
#  --validation_file '/home/etsuko/dialdoc/data/test_dev/doc2dial_dial_testdev.json'\
#  --model_name_or_path 'save/roberta-base/checkpoint-10000' \
#  --do_eval \
#  --version_2_with_negative \
#  --max_seq_length 512  \
#  --max_answer_length 50 \
#  --doc_stride 128  \
#  --output_dir save\
#  --per_device_eval_batch_size 8 \
#  --fp16 