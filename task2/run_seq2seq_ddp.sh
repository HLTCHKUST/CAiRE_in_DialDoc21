python -m torch.distributed.launch --nproc_per_node=2 --master_port=10003 finetune_trainer.py \
--cache_dir cache \
--output_dir save/bart-large-wow-hist3-sp \
--doc_mode sp \
--history_len 3 \
--num_train_epochs 20 \
--model_name_or_path [PATH TO WOW PRETRAINED MODEL] \
--early_stopping_patience 5 \
--learning_rate 3e-5 \
--adam_epsilon 1e-06 \
--do_train \
--do_eval \
--per_device_train_batch_size 5 \
--per_device_eval_batch_size 5 \
--overwrite_output_dir \
--adam_eps 1e-06 \
--max_source_length 300 \
--max_target_length 200 \
--val_max_target_length 50 \
--test_max_target_length 50 \
--task translation \
--warmup_steps 500 \
--evaluation_strategy epoch \
--load_best_model_at_end \
--predict_with_generate \
--save_total_limit 5 \
--metric_for_best_model eval_bleu \
--greater_is_better True \
--logging_steps 500 \
--sharded_ddp 



# # Without wow pre-training
# python -m torch.distributed.launch --nproc_per_node=2 --master_port=10003 finetune_trainer.py \
# --cache_dir cache \
# --output_dir save/bart-large-hist3-sp \
# --doc_mode sp \
# --history_len 3 \
# --num_train_epochs 20 \
# --model_name_or_path facebook/bart-large-cnn \
# --early_stopping_patience 5 \
# --learning_rate 3e-5 \
# --adam_epsilon 1e-06 \
# --do_train \
# --do_eval \
# --per_device_train_batch_size 5 \
# --per_device_eval_batch_size 5 \
# --overwrite_output_dir \
# --adam_eps 1e-06 \
# --max_source_length 300 \
# --max_target_length 200 \
# --val_max_target_length 50 \
# --test_max_target_length 50 \
# --task translation \
# --warmup_steps 500 \
# --evaluation_strategy epoch \
# --load_best_model_at_end \
# --predict_with_generate \
# --save_total_limit 5 \
# --metric_for_best_model eval_bleu \
# --greater_is_better True \
# --logging_steps 500 \
# --sharded_ddp 
