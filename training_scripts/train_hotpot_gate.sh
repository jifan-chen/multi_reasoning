# $1 cuda, $2 save dir
#OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$1 allennlp train experiments/hotpot_gate/noam_my_hotpot_legacy_multiprocess_full.json -s ./save/$2 --include-package my_library -f
CUDA_VISIBLE_DEVICES=$1 allennlp predict ./save/$2/model.tar.gz '../hotpot/dev/dev_distractor_chain.json' --output-file ./save/$2/$2_devpredict.json --batch-size 10 --cuda-device 0 --use-dataset-reader -o '{"model": {"output_att_scores": false}}' --predictor hotpot_predictor --include-package my_library --silent
python3 split_dev.py ../hotpot/dev/dev_distractor_coref_easy.json ../hotpot/dev/dev_distractor_coref_hard.json ./save/$2/$2_devpredict.json ./save/$2
