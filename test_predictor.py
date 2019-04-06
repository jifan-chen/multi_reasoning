import my_library
import argparse
import json, pickle
from allennlp.models.archival import load_archive
from my_library.predictors import HotpotPredictor


def test(predictor, data_path, tag):
    correct_count = 0
    total_count = 0
    for instance in predictor._dataset_reader.read(data_path):
        output = predictor.predict_instance(instance)
        total_count += 1
        # print(output['sent_labels'])
        # print(output['gate'])
        # print(output['gate'])
        for label, prob in zip(output['sent_labels'], output['gate']):
            if label == 1 and prob == 1:
                correct_count += 1
                break
    print('correct_count:', correct_count)
    print('total_count', total_count)
    metrics = predictor._model.get_metrics(reset=True)
    print(metrics)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='hotpot predictor test')
    parser.add_argument('model_path', type=str, help='path to the tgz model file')
    args = parser.parse_args()

    archive = load_archive(args.model_path, cuda_device=0)
    predictor = HotpotPredictor.from_archive(archive, 'hotpot_predictor')

    test(predictor, '/backup2/jfchen/data/squad/squad-dev-v1.1.json', 'Train')
    # test(predictor, '/scratch/cluster/jfchen/jason/multihopQA/hotpot/dev/dev_distractor.json', 'Validation')
