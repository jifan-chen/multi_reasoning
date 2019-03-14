import my_library
import argparse
import json, pickle
from allennlp.models.archival import load_archive
from my_library.predictors import HotpotPredictor
from tqdm import tqdm



def test(predictor, data_path):
    for instance in tqdm(predictor._dataset_reader.read(data_path)):
        output = predictor.predict_instance(instance)
    metrics = predictor._model.get_metrics(reset=True)
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='hotpot predictor test')
    parser.add_argument('model_path', type=str, help='path to the tgz model file')
    args = parser.parse_args()

    archive = load_archive(args.model_path, cuda_device=0)
    predictor = HotpotPredictor.from_archive(archive, 'hotpot_predictor')

    train_m = test(predictor, '/scratch/cluster/jfchen/jason/multihopQA/hotpot/test/test_10000_coref.json')
    val_m = test(predictor, '/scratch/cluster/jfchen/jason/multihopQA/hotpot/dev/dev_distractor_coref.json')
    print("{} Results: Avg F1 {} - Avg EM {}".format("Train", train_m['f1'], train_m['em']))
    print("{} Results: Avg F1 {} - Avg EM {}".format("Validation", val_m['f1'], val_m['em']))
