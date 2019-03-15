import my_library
import argparse
import json, pickle
from allennlp.models.archival import load_archive
from my_library.predictors import HotpotPredictor


def test(predictor, data_path, tag):
    for instance in predictor._dataset_reader.read(data_path):
        output = predictor.predict_instance(instance)
        print(output['sent_labels'])
        print(sum(output['gate']) / len(output['gate']))
        # print(output['gate'])
        for label, prob in zip(output['sent_labels'], output['gate']):
            if label == 1:
                print(label, prob)


    metrics = predictor._model.get_metrics(reset=True)
    print("{} Results: Avg F1 {} - Avg EM {}".format(tag, metrics['f1'], metrics['em']))
    print(metrics)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='hotpot predictor test')
    parser.add_argument('model_path', type=str, help='path to the tgz model file')
    args = parser.parse_args()

    archive = load_archive(args.model_path, cuda_device=0)
    predictor = HotpotPredictor.from_archive(archive, 'hotpot_predictor')

    test(predictor, '/backup2/jfchen/data/hotpot/dev/dev_100.json', 'Train')
    # test(predictor, '/scratch/cluster/jfchen/jason/multihopQA/hotpot/dev/dev_distractor.json', 'Validation')