# TODO: trainer.predict()
# look at task2/finetune_trainer.py

python run_qa.py \
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
