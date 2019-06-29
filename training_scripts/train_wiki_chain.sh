# $1 cuda, $2 save dir
#OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$1 allennlp train experiments/wikihop_chain/noam_hotpot_legacy_rl_multiprocess.json -s ./save/$2 --include-package my_library -f
CUDA_VISIBLE_DEVICES=$1 allennlp predict ./save/$2/model.tar.gz '../wikihop/dev/dev_chain.json' --output-file ./save/$2/$2_devpredict.json --batch-size 10 --cuda-device 0 --use-dataset-reader -o '{"model": {"output_att_scores": false, "span_gate": {"beam_size": 10}}}' --predictor hotpot_predictor --include-package my_library --silent
