#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @Time  : 2022/10/24 15:15
# @Author: lionel
import json
import os
from collections import OrderedDict
from random import choice

import torch
from torch.utils import data
from transformers import BertTokenizer

from models.casRel import MyLoss, CasRel


class DuieDataset(data.Dataset):
    def __init__(self, file_path, schema_data):
        self.data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                record = json.loads(line.strip(), encoding='utf-8')
                text = record['text']
                rels = []
                for spo in record['spo_list']:
                    features = dict()
                    predicate = spo['predicate']
                    features['predicate'] = schema_data['predicates'][predicate]
                    features['subject'] = spo['subject']
                    features['objects'] = list(spo['object'].values())
                    rels.append(features)
                self.data.append((text, rels))

    def __getitem__(self, item):
        text, rels = self.data[item]
        return text, rels

    def __len__(self):
        return len(self.data)


bert_model_path = '/tmp/chinese-roberta-wwm-ext'
tokenizer = BertTokenizer.from_pretrained(bert_model_path)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_start_end_index(word, text):
    word_chars = list(word)
    text_chars = list(text)

    start, end = 0, 0
    while end < len(text_chars) and start < len(word_chars):
        if text_chars[end] == word_chars[start]:
            start += 1
            end += 1
        else:
            end = end - start + 1
            start = 0
    if start == len(word_chars):
        return (end - len(word), end)
    return None


def build_vocab(vocab_path):
    vocab2id = OrderedDict()
    with open(vocab_path, 'r', encoding='utf-8') as f:
        for line in f:
            vocab2id[line.strip('\n')] = len(vocab2id)
    id2vocab = {val: key for key, val in vocab2id.items()}
    return vocab2id, id2vocab


vocab, _ = build_vocab(os.path.join(bert_model_path, 'vocab.txt'))


def collate_fn(batch):
    """

    :param batch:
    :return:
    """
    schema_data = json.load(open('../data/schema.json', 'r'))
    rel_nums = len(schema_data['predicates'])
    texts, rels = zip(*batch)

    batch_size = len(texts)
    seq_len = max([len(text) for text in texts]) + 2

    token_ids = torch.zeros((batch_size, seq_len), dtype=torch.int, device=device)
    token_type_ids = torch.zeros((batch_size, seq_len), dtype=torch.int, device=device)
    attention_mask = torch.zeros((batch_size, seq_len), dtype=torch.int, device=device)

    for index, text in enumerate(texts):
        token_ids[index][0] = vocab['[CLS]']
        attention_mask[index][0] = 1
        i = 0
        while i < len(text):
            token_ids[index][i + 1] = vocab.get(text[i], vocab['[UNK]'])
            attention_mask[index][i + 1] = 1
            i += 1
        token_ids[index][i + 1] = vocab['[SEP]']
        attention_mask[index][i + 1] = 1

    sub_heads = torch.zeros((batch_size, seq_len), dtype=torch.float, device=device)
    sub_tails = torch.zeros((batch_size, seq_len), dtype=torch.float, device=device)
    obj_heads = torch.zeros((batch_size, seq_len, rel_nums), dtype=torch.float, device=device)
    obj_tails = torch.zeros((batch_size, seq_len, rel_nums), dtype=torch.float, device=device)

    for index, rel in enumerate(rels):
        subjects = set()
        text = texts[index]
        for ele in rel:
            subject = ele['subject']
            position = get_start_end_index(subject, text)
            try:
                start, end = position
                sub_heads[index][start + 1] = 1.0  # 考虑bert预训练模型在句子首部添加[CLS]
                sub_tails[index][end] = 1.0  # re.finditer函数本身字符结尾（ele.end()）索引就多一位
                subjects.add(subject)
            except:
                print(subject, text)

        if subjects:
            random_subject = choice(list(subjects))  # 随机抽取一个subject， 进行关系抽取建模
            for ele in rel:
                subject = ele['subject']
                if subject != random_subject:
                    continue
                predicate = ele['predicate']
                for object in ele['objects']:
                    position = get_start_end_index(object, text)
                    start, end = position
                    obj_heads[index][start + 1][predicate] = 1.0  # 考虑bert预训练模型在句子首部添加[CLS]
                    obj_tails[index][end][predicate] = 1.0  # re.finditer函数本身字符结尾（ele.end()）索引就多一位

    return texts, token_ids, token_type_ids, attention_mask, sub_heads, sub_tails, obj_heads, obj_tails


def schema_process(schema_file_path):
    """

    :param schema_file_path:
    :return:
    """
    schema_data = dict()
    with open(schema_file_path, 'r', encoding='utf-8') as f:
        relations = dict()
        subject_types, object_types = [], []
        for line in f:
            ele = json.loads(line.strip(), encoding='utf-8')
            relations[ele['predicate']] = len(relations)
            for key, val in ele['object_type'].items():
                object_type = key + '_' + val
            subject_types.append(ele['subject_type'])
            object_types.append(object_type)
        schema_data['predicates'] = relations
        schema_data['subject_types'] = subject_types
        schema_data['object_types'] = object_types

    return schema_data


if __name__ == '__main__':
    schema_data = json.load(open('../data/schema.json', 'r'))
    train_file_path = '/tmp/DuIE2.0/duie_train.json/duie_train.json'
    # dev_file_path = '/tmp/DuIE2.0/duie_dev.json/duie_dev.json'
    # test_file_path = '/tmp/DuIE2.0/duie_sample.json/duie_sample.json'

    train_dataset = DuieDataset(train_file_path, schema_data)
    train_dataloader = data.DataLoader(train_dataset, shuffle=False, batch_size=10, collate_fn=collate_fn)
    count = 0
    for index, ele in enumerate(train_dataloader):
        texts, token_ids, token_type_ids, attention_mask, sub_heads, sub_tails, obj_heads, obj_tails = ele

        loss = MyLoss(attention_mask)

        model = CasRel(bert_model_path, num_relations=48, bert_dim=768)
        pred_sub_heads, pred_sub_tails, pred_obj_heads, pred_obj_tails = model(token_ids, token_type_ids,
                                                                               attention_mask,
                                                                               sub_heads, sub_tails)

        # print(loss.loss_fn(pred_sub_heads, sub_heads))
        # print(loss.loss_fn(pred_sub_tails, sub_tails))
        print(pred_obj_heads.size())
        print(obj_heads.size())
        print(loss.loss_fn(pred_obj_heads, obj_heads))
        print(loss.loss_fn(pred_obj_tails, obj_tails))
        # print(sub_tails)
        # print(obj_tails[0][7])
        # print(obj_heads[0][7])
        break

    import torch.nn.functional as F

    from torch.nn import BCELoss
