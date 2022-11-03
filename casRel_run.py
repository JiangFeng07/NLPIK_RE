#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @Time  : 2022/10/25 22:47
# @Author: lionel
import json
import os
import torch
from torch.utils import data
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup
from utils import build_vocab

from datasets.duie import DuieDataset, collate_fn
from models.casRel import MyLoss, CasRelBert

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def predict(texts, token_ids, token_type_ids, attention_mask, model, id2label, h_bar=0.5, t_bar=0.5):
    pred_sub_heads, pred_sub_tails = model.get_subs(token_ids, token_type_ids, attention_mask)
    sub_heads = torch.where(pred_sub_heads[0] > h_bar)[0]
    sub_tails = torch.where(pred_sub_tails[0] > t_bar)[0]
    subjects = set()
    for sub_head in sub_heads:
        sub_tail = sub_tails[sub_tails >= sub_head]
        if sub_tail:
            sub_tail = sub_tail[0]
            subject = texts[0][sub_head - 1:sub_tail]
            subjects.add((subject, int(sub_head), int(sub_tail)))
    spo_list = set()
    if subjects:
        sub_head_mapping = torch.zeros((len(subjects), 1, model.encoded_text.size(1)), dtype=torch.float,
                                       device=device)
        sub_tail_mapping = torch.zeros((len(subjects), 1, model.encoded_text.size(1)), dtype=torch.float,
                                       device=device)
        for subject_idx, subject in enumerate(subjects):
            sub_head_mapping[subject_idx][0][subject[1]] = 1
            sub_tail_mapping[subject_idx][0][subject[2]] = 1
        _, _, pred_obj_heads, pred_obj_tails = model(token_ids, token_type_ids, attention_mask,
                                                     sub_head_mapping, sub_tail_mapping)
        obj_heads, obj_tails = torch.where(pred_obj_heads > h_bar), torch.where(pred_obj_tails > t_bar)
        for subject_idx, subject in enumerate(subjects):
            for obj_head, rel_head in zip(*obj_heads[1:3]):
                for obj_tail, rel_tail in zip(*obj_tails[1:3]):
                    if obj_head <= obj_tail and rel_head == rel_tail:
                        rel = id2label[int(rel_head)]
                        obj = texts[0][int(obj_head - 1):int(obj_tail)]
                        spo_list.add((subject[0], rel, obj))
                        break
    return subjects, spo_list


def metric(dataloader, id2label):
    correct_num, predict_num, gold_num = 0, 0, 0
    with tqdm(total=len(dataloader), desc='模型验证进度条') as pbar:
        for batch in dataloader:
            texts, rels, token_ids, token_type_ids, attention_mask = batch[:5]
            _, pred_spo_list = predict(texts, token_ids, token_type_ids, attention_mask, model, id2label)
            spo_list = set()
            for ele in rels[0]:
                rel = id2label[ele['predicate']]
                sub = ele['subject']
                for obj in ele['objects']:
                    spo_list.add((sub, rel, obj))
            predict_num += len(pred_spo_list)
            gold_num += len(spo_list)
            correct_num += len(spo_list & pred_spo_list)
            pbar.update(1)

    print("correct_num: {:3d}, predict_num: {:3d}, gold_num: {:3d}".format(correct_num, predict_num, gold_num))
    precision = correct_num / (predict_num + 1e-10)
    recall = correct_num / (gold_num + 1e-10)
    f1_score = 2 * precision * recall / (precision + recall + 1e-10)
    print('f1: {:4.2f}, precision: {:4.2f}, recall: {:4.2f}'.format(f1_score, precision, recall))
    return precision, recall, f1_score


if __name__ == '__main__':
    bert_model_path = '/tmp/chinese-roberta-wwm-ext'

    vocab, _ = build_vocab(os.path.join(bert_model_path, 'vocab.txt'))

    schema_data = json.load(open('./data/schema.json', 'r'))
    id2label = {val: key for key, val in schema_data['predicates']}
    train_file_path = '/tmp/DuIE2.0/duie_train.json/duie_train.json'
    dev_file_path = '/tmp/DuIE2.0/duie_dev.json/duie_dev.json'
    # test_file_path = '/tmp/DuIE2.0/duie_sample.json/duie_sample.json'

    train_dataset = DuieDataset(train_file_path, schema_data)
    train_dataloader = data.DataLoader(train_dataset, shuffle=True, batch_size=10,
                                       collate_fn=lambda ele: collate_fn(ele, vocab, schema_data))

    dev_dataset = DuieDataset(dev_file_path, schema_data)
    dev_dataloader = data.DataLoader(dev_dataset, collate_fn=lambda ele: collate_fn(ele, vocab, schema_data))

    model = CasRelBert(bert_model_path, num_relations=len(schema_data['predicates']), bert_dim=768)
    optimizer = AdamW(model.parameters(), lr=1e-5, correct_bias=False)

    epochs = 10

    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    count = 0
    epoch = 1
    while epoch <= epochs:
        model.train()
        with tqdm(total=len(train_dataloader), desc='模型训练进度条') as pbar:
            for step, batch in train_dataloader:
                texts, rels, token_ids, token_type_ids, attention_mask, sub_heads, sub_tails, obj_heads, obj_tails = batch

                myLoss = MyLoss(attention_mask)

                optimizer.zero_grad()

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
                pbar.set_postfix({'loss': '{0:1.5f}'.format(float(loss))})
                pbar.update(1)
                loss.backward()
                optimizer.step()
                scheduler.step()

        with torch.no_grad():
            metric(dev_dataloader, id2label)
