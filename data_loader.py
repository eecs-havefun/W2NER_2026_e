import json
import os

import numpy as np
import prettytable as pt
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import AutoTokenizer

import utils

os.environ["TOKENIZERS_PARALLELISM"] = "false"

dis2idx = np.zeros((1000), dtype="int64")
dis2idx[1] = 1
dis2idx[2:] = 2
dis2idx[4:] = 3
dis2idx[8:] = 4
dis2idx[16:] = 5
dis2idx[32:] = 6
dis2idx[64:] = 7
dis2idx[128:] = 8
dis2idx[256:] = 9


class Vocabulary(object):
    PAD = '<pad>'
    UNK = '<unk>'
    SUC = '<suc>'

    def __init__(self):
        self.label2id = {self.PAD: 0, self.SUC: 1}
        self.id2label = {0: self.PAD, 1: self.SUC}

    def add_label(self, label):
        label = label.lower()
        if label not in self.label2id:
            self.label2id[label] = len(self.label2id)
            self.id2label[self.label2id[label]] = label
        assert label == self.id2label[self.label2id[label]]

    def __len__(self):
        return len(self.label2id)

    def label_to_id(self, label):
        label = label.lower()
        return self.label2id[label]

    def id_to_label(self, i):
        return self.id2label[i]


def collate_fn(data):
    bert_inputs, grid_labels, grid_mask2d, pieces2word, dist_inputs, sent_length, entity_text = map(list, zip(*data))

    max_tok = np.max(sent_length)
    sent_length = torch.LongTensor(sent_length)
    max_pie = np.max([x.shape[0] for x in bert_inputs])
    bert_inputs = pad_sequence(bert_inputs, True)
    batch_size = bert_inputs.size(0)

    def fill(src_data, new_data):
        for j, x in enumerate(src_data):
            new_data[j, :x.shape[0], :x.shape[1]] = x
        return new_data

    dis_mat = torch.zeros((batch_size, max_tok, max_tok), dtype=torch.long)
    dist_inputs = fill(dist_inputs, dis_mat)

    labels_mat = torch.zeros((batch_size, max_tok, max_tok), dtype=torch.long)
    grid_labels = fill(grid_labels, labels_mat)

    mask2d_mat = torch.zeros((batch_size, max_tok, max_tok), dtype=torch.bool)
    grid_mask2d = fill(grid_mask2d, mask2d_mat)

    sub_mat = torch.zeros((batch_size, max_tok, max_pie), dtype=torch.bool)
    pieces2word = fill(pieces2word, sub_mat)

    return bert_inputs, grid_labels, grid_mask2d, pieces2word, dist_inputs, sent_length, entity_text


class RelationDataset(Dataset):
    def __init__(self, bert_inputs, grid_labels, grid_mask2d, pieces2word, dist_inputs, sent_length, entity_text):
        self.bert_inputs = bert_inputs
        self.grid_labels = grid_labels
        self.grid_mask2d = grid_mask2d
        self.pieces2word = pieces2word
        self.dist_inputs = dist_inputs
        self.sent_length = sent_length
        self.entity_text = entity_text

    def __getitem__(self, item):
        return (
            torch.LongTensor(self.bert_inputs[item]),
            torch.LongTensor(self.grid_labels[item]),
            torch.LongTensor(self.grid_mask2d[item]),
            torch.LongTensor(self.pieces2word[item]),
            torch.LongTensor(self.dist_inputs[item]),
            self.sent_length[item],
            self.entity_text[item],
        )

    def __len__(self):
        return len(self.bert_inputs)


def normalize_instance_metadata(instance, fallback_sent_id):
    """
    统一补齐 predict/export 阶段会用到的元信息。
    不改变原始字段，只是尽量补出 doc_id / sent_id / sentence / ner。
    """
    normalized = dict(instance)

    sentence = normalized.get("sentence", [])
    ner = normalized.get("ner", [])

    normalized["sentence"] = sentence
    normalized["ner"] = ner

    if "sent_id" not in normalized:
        if "sentence_id" in normalized:
            normalized["sent_id"] = normalized["sentence_id"]
        elif "sid" in normalized:
            normalized["sent_id"] = normalized["sid"]
        else:
            normalized["sent_id"] = fallback_sent_id

    if "doc_id" not in normalized:
        for key in ["docid", "doc_key", "guid", "id"]:
            if key in normalized:
                normalized["doc_id"] = normalized[key]
                break
        else:
            normalized["doc_id"] = None

    return normalized


def process_bert(data, tokenizer, vocab):
    bert_inputs = []
    grid_labels = []
    grid_mask2d = []
    dist_inputs = []
    entity_text = []
    pieces2word = []
    sent_length = []
    aligned_data = []

    for raw_index, raw_instance in enumerate(data):
        instance = normalize_instance_metadata(raw_instance, fallback_sent_id=raw_index)

        if len(instance["sentence"]) == 0:
            continue

        tokens = [tokenizer.tokenize(word) for word in instance["sentence"]]
        pieces = [piece for piece_group in tokens for piece in piece_group]

        bert_token_ids = tokenizer.convert_tokens_to_ids(pieces)
        bert_token_ids = np.array(
            [tokenizer.cls_token_id] + bert_token_ids + [tokenizer.sep_token_id],
            dtype=np.int64
        )

        length = len(instance["sentence"])
        grid = np.zeros((length, length), dtype=np.int64)
        piece_map = np.zeros((length, len(bert_token_ids)), dtype=np.bool_)
        dist = np.zeros((length, length), dtype=np.int64)
        mask2d = np.ones((length, length), dtype=np.bool_)

        start = 0
        for token_idx, token_pieces in enumerate(tokens):
            if len(token_pieces) == 0:
                continue
            piece_span = list(range(start, start + len(token_pieces)))
            piece_map[token_idx, piece_span[0] + 1: piece_span[-1] + 2] = 1
            start += len(token_pieces)

        for k in range(length):
            dist[k, :] += k
            dist[:, k] -= k

        for i in range(length):
            for j in range(length):
                if dist[i, j] < 0:
                    dist[i, j] = dis2idx[-dist[i, j]] + 9
                else:
                    dist[i, j] = dis2idx[dist[i, j]]

        dist[dist == 0] = 19

        for entity in instance["ner"]:
            index = entity["index"]
            for i in range(len(index) - 1):
                grid[index[i], index[i + 1]] = 1
            grid[index[-1], index[0]] = vocab.label_to_id(entity["type"])

        entity_set = set(
            [
                utils.convert_index_to_text(
                    e["index"],
                    vocab.label_to_id(e["type"])
                )
                for e in instance["ner"]
            ]
        )

        sent_length.append(length)
        bert_inputs.append(bert_token_ids)
        grid_labels.append(grid)
        grid_mask2d.append(mask2d)
        dist_inputs.append(dist)
        pieces2word.append(piece_map)
        entity_text.append(entity_set)
        aligned_data.append(instance)
        ###process_bert_output_aligned_checker
        assert len(aligned_data) == len(bert_inputs) == len(grid_labels) == len(grid_mask2d) == \
       len(dist_inputs) == len(pieces2word) == len(sent_length) == len(entity_text), (
    "process_bert outputs are misaligned"
)

    return (
        bert_inputs,
        grid_labels,
        grid_mask2d,
        pieces2word,
        dist_inputs,
        sent_length,
        entity_text,
    ), aligned_data


def fill_vocab(vocab, dataset):
    entity_num = 0
    for instance in dataset:
        for entity in instance["ner"]:
            vocab.add_label(entity["type"])
        entity_num += len(instance["ner"])
    return entity_num


def load_data_bert(config):
    import os
    
    # 使用配置中的数据路径
    train_path = os.path.join(config.data_root, config.dataset, "train.json")
    dev_path = os.path.join(config.data_root, config.dataset, "dev.json")
    test_path = os.path.join(config.data_root, config.dataset, "test.json")
    
    with open(train_path, "r", encoding="utf-8") as f:
        train_data = json.load(f)
    with open(dev_path, "r", encoding="utf-8") as f:
        dev_data = json.load(f)
    with open(test_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(config.bert_name, cache_dir=config.cache_dir)
    vocab = Vocabulary()

    train_ent_num = fill_vocab(vocab, train_data)
    dev_ent_num = fill_vocab(vocab, dev_data)
    test_ent_num = fill_vocab(vocab, test_data)

    table = pt.PrettyTable([config.dataset, "sentences", "entities"])
    table.add_row(["train", len(train_data), train_ent_num])
    table.add_row(["dev", len(dev_data), dev_ent_num])
    table.add_row(["test", len(test_data), test_ent_num])
    config.logger.info("\n{}".format(table))

    config.label_num = len(vocab.label2id)
    config.vocab = vocab

    train_features, train_aligned_data = process_bert(train_data, tokenizer, vocab)
    dev_features, dev_aligned_data = process_bert(dev_data, tokenizer, vocab)
    test_features, test_aligned_data = process_bert(test_data, tokenizer, vocab)

    # 对齐诊断：predict/export 阶段用的 ori_data 应与 dataset 中真实样本一一对应
    if len(train_aligned_data) != len(train_data):
        config.logger.info(
            "train: filtered {} empty sentences for aligned export metadata.".format(
                len(train_data) - len(train_aligned_data)
            )
        )
    if len(dev_aligned_data) != len(dev_data):
        config.logger.info(
            "dev: filtered {} empty sentences for aligned export metadata.".format(
                len(dev_data) - len(dev_aligned_data)
            )
        )
    if len(test_aligned_data) != len(test_data):
        config.logger.info(
            "test: filtered {} empty sentences for aligned export metadata.".format(
                len(test_data) - len(test_aligned_data)
            )
        )

    train_dataset = RelationDataset(*train_features)
    dev_dataset = RelationDataset(*dev_features)
    test_dataset = RelationDataset(*test_features)
    ###dataset_checker
    assert len(train_dataset) == len(train_data), "train dataset / ori_data misaligned"
    assert len(dev_dataset) == len(dev_data), "dev dataset / ori_data misaligned"
    assert len(test_dataset) == len(test_data), "test dataset / ori_data misaligned"

    return (
        train_dataset,
        dev_dataset,
        test_dataset,
    ), (
        train_aligned_data,
        dev_aligned_data,
        test_aligned_data,
    )
