python -m torch.distributed.launch --nproc_per_node=1 --master_port=10002 finetune_trainer.py \
--cache_dir cache \
--ids_path cache/test_ids.txt \
--preds_path ../results/test3_100_sp_ensemble_preds_yesno.json \
--output_dir save/bart-large-wow-cont-3/checkpoint-109990/test_100 \
--doc_mode sp \
--model_name_or_path save/bart-large-wow-cont-3/checkpoint-109990 \
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
# --predict_eval \