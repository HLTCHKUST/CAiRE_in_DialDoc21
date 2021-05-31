files=$(ls ../results/test3_100)
for filename in $files
do
python postprocess_prediction.py --task split --prediction_file ../results/test3_100/$filename \
--output_file ../results/test3_100_sp/$filename \
--split test --threshold 0.1 --save_span
done

# sh do_ensemble.sh ../results/sp_testdev3/$filename

# mkdir ../results/val3_sp_preds

# mv ../results/val3_sp/*preds* ../results/val3_sp_preds/

# files=$(ls ../results/sp_testdev3_preds)
# for filename in $files
# do
# python postprocess_prediction.py --split testdev --prediction_file ../results/sp_testdev3_preds/$filename --save --task patch
# done