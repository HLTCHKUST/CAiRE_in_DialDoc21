# CAiRE in DialDoc21

## Install environment
```
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## load dialdoc dataset
```
datasets = load_dataset("path/to/dialdoc.py", "doc2dial_rc")
```

## Task 1 
### Data postprocessing
- Convert
```
python postprocess_prediction.py --task convert --prediction_file save/MODELPATH/testdev/predictions.json --output_file save/MODELPATH/testdev/predictions_submit.json --source_file ../data/dialdoc21-sharedtask-phase1/test_subtask1_phase1_ids.json
```

- Split
If calculating the metrics is needed, add `--do_eval`.
```
python postprocess_prediction.py --task split --folder /path/to/the/preds/folder --split [validation/devtest/test] --do_eval --do_filter --source_file ../data/dialdoc21-sharedtask-phase1/test_subtask1_phase1_ids.json
```

## Task 2
### Data preprocessing
```
python seq2seq_utils.py --split testdev --doc_mode grounding --data_dir data/grounding --sample_file ../data/dialdoc21-sharedtask-phase1/test_subtask2_phase1_ids.json --preds_file ../save/task1/roberta-large/test_dev/sp_predictions.json
```

### Data postprocessing
```
python seq2seq_utils.py --task output --split testdev --doc_mode grounding --data_dir data/grounding --preds_file save/bart-large-cnn-grounding/test_generations.txt
```

## Evaluation
### Subtask 1


### Subtask 2
