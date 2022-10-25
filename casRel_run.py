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

if __name__ == '__main__':
    bert_model_path = '/tmp/chinese-roberta-wwm-ext'

    schema_data = json.load(open('./data/schema.json', 'r'))
    vocab, _ = build_vocab(os.path.join(bert_model_path, 'vocab.txt'))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
