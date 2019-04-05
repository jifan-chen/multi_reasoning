import my_library
import argparse
import json, pickle
from allennlp.models.archival import load_archive
from my_library.predictors import HotpotPredictor
from tqdm import tqdm
import sys


'''
def test(predictor, data_path):
    for instance in tqdm(predictor._dataset_reader.read(data_path)):
        output = predictor.predict_instance(instance)
    metrics = predictor._model.get_metrics(reset=True)
    return metrics
'''
def test(predictor, dataset):
    num = len(predictor.demo_dataset[dataset])
    other_metrics = {"evd_p": 0.,
                     "evd_r": 0.,
                     "evd_f1": 0.,
                     "f1": 0.}
    for i in range(num):
        output = predictor.predict_json({"dataset": dataset, "th": 0.2, "instance_idx": i})
        other_metrics['evd_p'] += output['evd_measure']['prec']
        other_metrics['evd_r'] += output['evd_measure']['recl']
        other_metrics['evd_f1'] += output['evd_measure']['f1']
        other_metrics['f1'] += output['f1']
        print(dataset, i,
              'evd_p:', output['evd_measure']['prec'],
              'evd_r:', output['evd_measure']['recl'],
              'evd_f1:', output['evd_measure']['f1'],
              'f1:', output['f1'])
        sys.stdout.flush()
    for k in other_metrics:
        other_metrics[k] /= num
    metrics = predictor._model.get_metrics(reset=True)
    return metrics, other_metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='hotpot predictor test')
    parser.add_argument('model_path', type=str, help='path to the tgz model file')
    args = parser.parse_args()

    archive = load_archive(args.model_path, cuda_device=0)
    predictor = HotpotPredictor.from_archive(archive, 'hotpot_predictor')

    '''
    train_m = test(predictor, '/scratch/cluster/jfchen/jason/multihopQA/hotpot/test/test_10000_coref.json')
    val_m = test(predictor, '/scratch/cluster/jfchen/jason/multihopQA/hotpot/dev/dev_distractor_coref.json')
    print("{} Results: Avg F1 {} - Avg EM {}".format("Train", train_m['f1'], train_m['em']))
    print("{} Results: Avg F1 {} - Avg EM {}".format("Validation", val_m['f1'], val_m['em']))
    '''
    train_m, train_other = test(predictor, 'train')
    val_m, val_other = test(predictor, 'dev')
    print(train_m)
    print(train_other)
    print(val_m)
    print(val_other)
