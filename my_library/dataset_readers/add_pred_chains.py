import os
import json
import glob
import argparse


def add_pred_chains(hotpot_path, pred_path, output_dir):
    pred_results = []
    with open(pred_path, 'r') as f:
        for line in f:
            pred_results.append(json.loads(line))
    res_id2idx = {d['_id']: i for i, d in enumerate(pred_results)}

    for fn in sorted(glob.glob(hotpot_path)):
        with open(fn, 'r') as f:
            orig_data = json.load(f)
        print("load data of size %d from %s" % (len(orig_data), fn))
    
        for d in orig_data:
            assert not 'pred_chains' in d
            d['pred_chains'] = pred_results[res_id2idx[d['_id']]]['pred_chains']

        short_fn = fn.split('/')[-1]
        output_path = os.path.join(output_dir, 'pred_'+short_fn)
        with open(output_path, 'w') as f:
            json.dump(orig_data, f, sort_keys=True)
        print("save data of size %d to %s" % (len(orig_data), output_path))


if __name__ == '__main__':
    parser = parser = argparse.ArgumentParser(description='add the chain predictions by beam search into hotpot instances')
    parser.add_argument('hotpot_path', type=str, help='path to file storing hotpot instances')
    parser.add_argument('pred_path', type=str, help='path to file storing chain predictions')
    parser.add_argument('output_dir', type=str, help='the directory to store the output')
    args = parser.parse_args()

    add_pred_chains(args.hotpot_path, args.pred_path, args.output_dir)
