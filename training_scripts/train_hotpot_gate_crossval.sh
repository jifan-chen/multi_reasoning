# $1 cuda, $2 save dir, $3 fold
#mkdir -p ./save/$2
#OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$1 allennlp train experiments/hotpot_gate/noam_my_hotpot_legacy_multiprocess_crossval$3.json -s ./save/$2/$2train$3valdev/ --include-package my_library -f
mkdir -p ./save/$2/train_pred_chain
CUDA_VISIBLE_DEVICES=$1 allennlp predict ./save/$2/$2train$3valdev/model.tar.gz "../hotpot/train_chain_2fold/fold$((1 - $3))/train*.json" --output-file ./save/$2/train_pred_chain/predict_trainfold$3evalfold$((1 - $3)).json --batch-size 10 --cuda-device 0 --use-dataset-reader -o '{"model": {"output_att_scores": false}}' --predictor hotpot_predictor --include-package my_library --silent
