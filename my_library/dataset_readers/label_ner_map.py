import json
import argparse
from tqdm import tqdm
from allennlp.predictors.predictor import Predictor
from allennlp.common.util import sanitize
from allennlp.models.archival import load_archive
from allennlp.data.tokenizers.sentence_splitter import SpacySentenceSplitter


class AllenNER:
    def __init__(self, path):
        archive = load_archive(path, cuda_device=0)
        self.predictor = Predictor.from_archive(archive)

    def predict_raw(self, sentence):
        instance = self.predictor._json_to_instance({'sentence': sentence})
        output = self.predictor._model.forward_on_instance(instance)
        return sanitize(output)

    def predict_batch_raw(self, sentences):
        '''
        outputs = [out]
        out = {'words': ...
               'pos': ...
               'predicted_heads': ...
               'predicted_dependencies': ...}
        '''
        batch_json = [{'sentence': sent} for sent in sentences]
        instances = self.predictor._batch_json_to_instances(batch_json)
        outputs = self.predictor._model.forward_on_instances(instances)
        return sanitize(outputs)

    def predict_tokens(self, tokens):
        instance = self.predictor._dataset_reader.text_to_instance(tokens[0], tokens[1])
        output = self.predictor._model.forward_on_instance(instance)
        return sanitize(output)

    def predict_batch_tokens(self, tokens_list):
        instances = [self.predictor._dataset_reader.text_to_instance(toks, tags) for toks, tags in tokens_list]
        outputs = self.predictor._model.forward_on_instances(instances)
        return sanitize(outputs)


def entity_extraction_(words, tags):
    in_entity = False
    entities = []
    phrase = []
    for w, t in zip(words, tags):
        if t != 'O':
            phrase.append(w)
            in_entity = True
        elif t == 'O' and in_entity:
            in_entity = False
            entities.append(" ".join(phrase))
            phrase = []
    return entities


def entity_extraction_hotpot(args):
    predictor_conll = AllenNER(
        "https://s3-us-west-2.amazonaws.com/allennlp/models/ner-model-2018.12.18.tar.gz")
    predictor_onto_note = \
        AllenNER("https://s3-us-west-2.amazonaws.com/allennlp/models/fine-grained-ner-model-elmo-2018.12.21.tar.gz")
    with open(args.path, 'r') as f:
        data = json.load(f)
    for d in tqdm(data):
        golden_ners = []
        parsing_info = []
        question = d['question'].strip().replace("\n", "")
        output1 = predictor_conll.predict_raw(question)
        output2 = predictor_onto_note.predict_raw(question)
        question_entities1 = entity_extraction_(output1['words'], output1['tags'])
        question_entities2 = entity_extraction_(output2['words'], output2['tags'])
        question_entities = set(question_entities1).union(set(question_entities2))
        # print(question_entities)
        for title, para in d['context']:
            para_ners = [title, []]
            outputs_conll = predictor_conll.predict_batch_raw(para)
            outputs_onto_note = predictor_onto_note.predict_batch_raw(para)
            for out1, out2 in zip(outputs_conll, outputs_onto_note):
                entities1 = entity_extraction_(out1['words'], out1['tags'])
                entities2 = entity_extraction_(out2['words'], out2['tags'])
                entities = set(entities1).union(set(entities2))
                # print(entities)
                para_ners[1].append(list(entities))
            golden_ners.append(para_ners)
            parsing_info.append([title, outputs_conll])

        d['question_entities'] = list(question_entities)
        d['ners'] = golden_ners

    with open(args.output, 'w') as f:
        json.dump(data, f)


def entity_extraction_wikihop(args):
    predictor_conll = AllenNER(
        "https://s3-us-west-2.amazonaws.com/allennlp/models/ner-model-2018.12.18.tar.gz")
    predictor_onto_note = \
        AllenNER("https://s3-us-west-2.amazonaws.com/allennlp/models/fine-grained-ner-model-elmo-2018.12.21.tar.gz")
    sentence_splitter = SpacySentenceSplitter(rule_based=True)
    with open(args.path, 'r') as f:
        data = json.load(f)
    for d in tqdm(data):
        golden_ners = []
        passage = []
        question = d['query'].strip().replace("\n", "")
        question_entity = " ".join(question.split()[1:])
        question = " ".join(question.split("_"))
        for para in d['supports']:
            sentences = sentence_splitter.split_sentences(para)
            para_ners = []
            outputs_conll = predictor_conll.predict_batch_raw(sentences)
            outputs_onto_note = predictor_onto_note.predict_batch_raw(sentences)
            for out1, out2 in zip(outputs_conll, outputs_onto_note):
                entities1 = entity_extraction_(out1['words'], out1['tags'])
                entities2 = entity_extraction_(out2['words'], out2['tags'])
                entities = set(entities1).union(set(entities2))
                # print(entities)
                para_ners.append(list(entities))
            golden_ners.append(para_ners)
            passage.append(sentences)
            # parsing_info.append([title, outputs_conll])
        # print(question)
        # print(question_entity)
        # input()
        d['supports'] = passage
        d['question_entities'] = [question_entity]
        d['ners'] = golden_ners
        d['query'] = question
        # input()
    with open(args.output, 'w') as f:
        json.dump(data, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='label golden dependency heads map for hotpot dataset')
    parser.add_argument('--path', help='path to hotpot dataset')
    parser.add_argument('--output', help='path to dep-labeled hotpot dataset')
    # parser.add_argument('ner_output', help='path to result of dependency parsing')
    parser.add_argument('--num', type=int, help='number of data to evaluate', default=-1)
    parser.add_argument('--draw', action='store_true', help='draw dep parsing tree', default=False)
    args = parser.parse_args()
    # entity_extraction_hotpot(args)
    entity_extraction_wikihop(args)



