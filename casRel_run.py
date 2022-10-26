#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @Time  : 2022/10/25 22:47
# @Author: lionel
import json
import os
import torch
from torch.utils import data
from transformers import AdamW, get_linear_schedule_with_warmup
from utils import build_vocab

from datasets.duie import DuieDataset, collate_fn
from models.casRel import MyLoss, CasRel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def tokenizer(texts, vocab):
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
    return token_ids, token_type_ids, attention_mask


def predict(texts, model, id2label, vocab, h_bar=0.5, t_bar=0.5):
    token_ids, token_type_ids, attention_mask = tokenizer(texts, vocab)
    pred_sub_heads, pred_sub_tails = model.get_subs(token_ids, token_type_ids, attention_mask)
    sub_heads = torch.where(pred_sub_heads[0] > h_bar)[0]
    sub_tails = torch.where(pred_sub_tails[0] > t_bar)[0]
    subjects = []
    for sub_head in sub_heads:
        sub_tail = sub_tails[sub_tails >= sub_head]
        if sub_tail:
            sub_tail = sub_tail[0]
            subject = texts[0][sub_head - 1:sub_tail]
            subjects.append((subject, int(sub_head), int(sub_tail)))
    spo_list = []
    if subjects:
        sub_head_mapping = torch.zeros((len(subjects), 1, model.encoded_text.size(1)), dtype=torch.float,
                                       device=device)
        sub_tail_mapping = torch.zeros((len(subjects), 1, model.encoded_text.size(1)), dtype=torch.float,
                                       device=device)
        for subject_idx, subject in enumerate(subjects):
            sub_head_mapping[subject_idx][0][subject[1]] = 1
            sub_tail_mapping[subject_idx][0][subject[2]] = 1
            _, _, pred_obj_heads, pred_obj_tails = model(token_ids, token_type_ids, attention_mask,
                                                         sub_head_mapping[subject_idx],
                                                         sub_tail_mapping[subject_idx])
            obj_heads, obj_tails = torch.where(pred_obj_heads > h_bar), torch.where(pred_obj_tails > t_bar)
            for obj_head, rel_head in zip(*obj_heads[1:3]):
                for obj_tail, rel_tail in zip(*obj_tails[1:3]):
                    if obj_head <= obj_tail and rel_head == rel_tail:
                        rel = id2label[int(rel_head)]
                        obj = texts[0][int(obj_head - 1):int(obj_tail)]
                        spo_list.append((subject[0], rel, obj))
                        break
    return subjects, spo_list


if __name__ == '__main__':
    bert_model_path = '/tmp/chinese-roberta-wwm-ext'

    vocab, _ = build_vocab(os.path.join(bert_model_path, 'vocab.txt'))

    schema_data = json.load(open('./data/schema.json', 'r'))
    train_file_path = '/tmp/DuIE2.0/duie_train.json/duie_train.json'
    dev_file_path = '/tmp/DuIE2.0/duie_dev.json/duie_dev.json'
    # test_file_path = '/tmp/DuIE2.0/duie_sample.json/duie_sample.json'

    train_dataset = DuieDataset(train_file_path, schema_data)
    train_dataloader = data.DataLoader(train_dataset, shuffle=True, batch_size=10,
                                       collate_fn=lambda ele: collate_fn(ele, vocab, schema_data))

    dev_dataset = DuieDataset(dev_file_path, schema_data)
    dev_dataloader = data.DataLoader(dev_dataset, batch_size=10,
                                     collate_fn=lambda ele: collate_fn(ele, vocab, schema_data))

    model = CasRel(bert_model_path, num_relations=len(schema_data['predicates']), bert_dim=768)
    optimizer = AdamW(model.parameters(), lr=1e-5, correct_bias=False)

    epochs = 10

    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    count = 0
    epoch = 1
    while epoch <= epochs:
        for batch in train_dataloader:
            texts, token_ids, token_type_ids, attention_mask, sub_heads, sub_tails, obj_heads, obj_tails = batch

            myLoss = MyLoss(attention_mask)

            pred_sub_heads, pred_sub_tails, pred_obj_heads, pred_obj_tails = model(token_ids, token_type_ids,
                                                                                   attention_mask,
                                                                                   sub_heads, sub_tails)
            predict_label, gold_label = dict(), dict()
            predict_label['pred_sub_heads'] = pred_sub_heads
            predict_label['pred_sub_tails'] = pred_sub_tails
            predict_label['pred_obj_heads'] = pred_obj_heads
            predict_label['pred_obj_tails'] = pred_obj_tails

            gold_label['sub_heads'] = sub_heads
            gold_label['sub_tails'] = sub_tails
            gold_label['obj_heads'] = obj_heads
            gold_label['obj_tails'] = obj_tails

            loss = myLoss(predict_label, gold_label)
            print(loss)
            # loss.backward()
            # scheduler.step()
            # optimizer.step()
