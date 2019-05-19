import json
import argparse


def split_data(original_path, id_path, output_path):
    id_file = json.load(open(id_path))
    original_file = json.load(open(original_path))
    output_file = []
    for instance in original_file:
        id = instance['_id']
        if id in id_file:
            output_file.append(instance)
    json.dump(output_file, open(output_path, 'w'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='hotpot predictor test')
    parser.add_argument('--original_path', type=str, help='path to the original file')
    parser.add_argument('--id_path', type=str, help='path to the id data file')
    parser.add_argument('--output_path', type=str, help='path to the output file')
    args = parser.parse_args()
    split_data(args.original_path, args.id_path, args.output_path)

