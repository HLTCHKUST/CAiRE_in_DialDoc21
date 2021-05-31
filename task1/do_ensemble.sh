# Given the prediction span, return the corresponding text in the context
python run_qa.py \
 --dataset_name '../utils/dialdoc/dialdoc.py' \
 --dataset_config_name doc2dial_rc_test \
 --model_name_or_path roberta-large \
 --max_seq_length 512  \
 --max_answer_length 100 \
 --doc_stride 128  \
 --cache_dir cache \
 --output_dir [OUTPUT FOLDER] \
 --do_ensemble \
 --ensemble_file_path test_sp_ensemble.json