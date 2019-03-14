import json


def split_dataset(input_path, output_path):
    interval = 2000
    input_data = json.load(open(input_path, 'r'))
    output_data = []
    file_count = 0
    for i in range(len(input_data)):
        output_data.append(input_data[i])
        if (i + 1) % interval == 0:
            json.dump(output_data, open("{}{}.json".format(output_path, file_count), 'w'))
            file_count += 1
            output_data = []
    json.dump(output_data, open("{}{}.json".format(output_path, file_count), 'w'))


def create_test_data(input_path, output_path, data_number=10000):
    start = 0
    input_data = json.load(open(input_path, 'r'))
    test_data = []
    count = 0
    for i in range(start, start + data_number):
        count += 1
        test_data.append(input_data[i])
    print(len(test_data))
    json.dump(test_data, open(output_path, 'w'))


if __name__ == '__main__':
    # original_path = '/backup2/jfchen/data/hotpot/train/train_coref.json'
    # output_url = '/backup2/jfchen/data/hotpot/test/hotpot_test_10000.json'
    # create_test_data(original_path, output_url)
    original_path = '/backup2/jfchen/data/hotpot/test/hotpot_test_100.json'
    output_url = '/backup2/jfchen/data/hotpot/test/test_100/test'
    split_dataset(original_path, output_url)

