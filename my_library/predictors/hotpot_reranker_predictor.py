import argparse
import json
from allennlp.models.archival import load_archive
# from my_library.predictors import HotpotPredictor
from allennlp.models.archival import Archive, load_archive
from allennlp.common.checks import ConfigurationError
from allennlp.data import DatasetReader, Instance
# from my_library.models.hotpot_bert_reranker import BidirectionalAttentionFlow


def test(model, dataset_reader, data_path, tag):
    total_count = 0
    correct_labels = []
    wrong_labels = []

    for instance in dataset_reader.read(data_path):
        output = model.forward_on_instance(instance)
        total_count += 1
        print(output['score'])
        print(output['_id'])
        print(output['pred_chains'])
        input()

        # print(output['loss'])
        # print(output['sent_labels'])
        # print(output['gate'])
        # print(output['gate'])
        # for label, prob in zip(output['sent_labels'], output['gate']):
        #     if label == 1 and prob == 1:
        #         correct_count += 1
        #         break
    json.dump(correct_labels, open("/backup2/jfchen/data/hotpot/dev/dev_easy_ids.json", 'w'))
    json.dump(wrong_labels, open("/backup2/jfchen/data/hotpot/dev/dev_hard_ids.json", 'w'))
    print('correct_labels:', correct_labels)
    print('wrong_labels', wrong_labels)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='hotpot predictor test')
    parser.add_argument('--model_path', type=str, help='path to the tgz model file')
    parser.add_argument('--data_path', type=str, help='path to the input data file')
    args = parser.parse_args()

    archive = load_archive(args.model_path, cuda_device=0)
    config = archive.config.duplicate()
    dataset_reader_params = config["dataset_reader"]
    dataset_reader = DatasetReader.from_params(dataset_reader_params)
    model = archive.model
    model.eval()

    # predictor = HotpotPredictor.from_archive(archive, 'hotpot_predictor')

    test(model, dataset_reader, args.data_path, 'Train')
    # test(predictor, '/scratch/cluster/jfchen/jason/multihopQA/hotpot/dev/dev_distractor.json', 'Validation')
