python -m torch.distributed.launch --nproc_per_node=1 --master_port=10002 finetune_trainer.py \
--cache_dir cache \
--ids_path cache/test_ids.txt \
--preds_path ../results/[PATH TO TASK1 PREDICTIONS] \
--output_dir [PATH TO YOUR MODEL]/test \
--doc_mode sp \
--model_name_or_path [PATH TO YOUR MODEL] \
--stage test \
--do_predict \
--per_device_eval_batch_size 5 \
--overwrite_output_dir \
--max_source_length 300 \
--max_target_length 200 \
--val_max_target_length 200 \
--test_max_target_length 200 \
--task translation \
--predict_with_generate \
--logging_steps 1000 

# # Predict on validation set
# python -m torch.distributed.launch --nproc_per_node=1 --master_port=10002 finetune_trainer.py \
# --cache_dir cache \
# --ids_path cache/val_ids.txt \
# --preds_path ../results/[PATH TO TASK1 PREDICTIONS] \
# --output_dir [PATH TO YOUR MODEL]/test \
# --doc_mode sp \
# --model_name_or_path [PATH TO YOUR MODEL] \
# --do_predict \
# --per_device_eval_batch_size 5 \
# --overwrite_output_dir \
# --max_source_length 300 \
# --max_target_length 200 \
# --val_max_target_length 200 \
# --test_max_target_length 200 \
# --task translation \
# --predict_with_generate \
# --logging_steps 1000 \
# --predict_eval \

# # Predict on testdev set
# python -m torch.distributed.launch --nproc_per_node=1 --master_port=10002 finetune_trainer.py \
# --cache_dir cache \
# --ids_path cache/val_ids.txt \
# --preds_path ../results/[PATH TO TASK1 PREDICTIONS] \
# --output_dir [PATH TO YOUR MODEL]/test \
# --doc_mode sp \
# --model_name_or_path [PATH TO YOUR MODEL] \
# --stage testdev \
# --do_predict \
# --per_device_eval_batch_size 5 \
# --overwrite_output_dir \
# --max_source_length 300 \
# --max_target_length 200 \
# --val_max_target_length 200 \
# --test_max_target_length 200 \
# --task translation \
# --predict_with_generate \
# --logging_steps 1000 