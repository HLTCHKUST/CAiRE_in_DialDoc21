# Subtask 1
## save predictions
python postprocess_prediction.py --task convert --prediction_file save/roberta-large/testdev-100/predictions.json --output_file save/roberta-large/testdev-100/predictions_submit.json --source_file ../data/dialdoc21-sharedtask-phase1/test_subtask1_phase1_ids.json

python postprocess_prediction.py --task split --prediction_file save/roberta-large/testdev-100/positions.json --split testdev --do_filter --source_file ../data/dialdoc21-sharedtask-phase1/testdev_subtask1_phase1_ids.json --threshold 0.1 --save 

python postprocess_prediction.py --task split --prediction_file save/roberta-large/val-100/positions.json --split validation --threshold 0.1 --save

### test phase
python postprocess_prediction.py --task split --prediction_file ../results/test1_ensemble.json --split test --do_filter --source_file ../dataset/doc2dial/v1.0.1/test_phase/test_subtask1_phase2_ids.json --threshold 0.1 --save

## save spans
python postprocess_prediction.py --task split --prediction_file save/roberta-large-mrqa-continue-seed909/checkpoint-4832/testdev/positions.json --output_file ../results/sp_testdev2/roberta-large-mrqa-continue-seed909-4832_testdev_positions.json --split testdev --threshold 0.1 --save_span

python postprocess_prediction.py --task split --prediction_file save/roberta-large-mrqa-dcq/test/positions.json --output_file ../results/sp_test/roberta-large-mrqa-dcq_test_positions.json --split test --threshold 0.1 --save_span

# Subtask 2

python seq2seq_utils.py --split testdev --doc_mode grounding --data_dir data/grounding --sample_file ../data/dialdoc21-sharedtask-phase1/test_subtask2_phase1_ids.json --preds_file ../task1/save/roberta-large-coqa-single/testdev/predictions_submit.json




# wow pretraining
CUDA_VISIBLE_DEVICES=4 python finetune_task.py --use_st --dataset wow --data_dir ./data --save_path dialdoc --tok gpt2 --pretrained_model gpt2 --model_name_or_path "" --max_context_length 256 --max_length 256 --gradient_accumulation_steps 1 --bsz 16 --eval_bsz 32 --epoch 50 --lr 1e-5 --warmup_steps 0 --exp wow-history-kn-gpt-all -hic -kic --lm

CUDA_VISIBLE_DEVICES=5 python finetune_task.py --use_st --dataset wow --data_dir ./data --save_path dialdoc --tok gpt2 --pretrained_model microsoft/DialoGPT-small --model_name_or_path "" --max_context_length 256 --max_length 256 --gradient_accumulation_steps 1 --bsz 8 --eval_bsz 16 --epoch 50 --lr 1e-5 --warmup_steps 0 --exp wow-history-kn-dialogpt-all -hic -kic --lm

CUDA_VISIBLE_DEVICES=4 python finetune_task.py --model_type seq2seq --use_st --dataset wow --data_dir ./data --save_path dialdoc --tok facebook/bart-base --pretrained_model facebook/bart-base --model_name_or_path "" --max_context_length 256 --max_length 256 --gradient_accumulation_steps 4 --bsz 4 --eval_bsz 16 --epoch 50 --lr 1e-5 --warmup_steps 0 --exp wow-history-kn-bart-base-all -hic -kic --lm

CUDA_VISIBLE_DEVICES=6 python finetune_task.py --model_type seq2seq --use_st --dataset wow --data_dir ./data --save_path dialdoc --tok facebook/bart-large --pretrained_model facebook/bart-large --model_name_or_path "" --max_context_length 256 --max_length 256 --gradient_accumulation_steps 16 --bsz 1 --eval_bsz 2 --epoch 50 --lr 1e-5 --warmup_steps 0 --exp wow-history-kn-bart-large-all -hic -kic --lm
