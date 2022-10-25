#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @Time  : 2022/10/24 09:44
# @Author: lionel
import os

import torch
from torch import nn

from transformers import BertModel, BertTokenizer

from datasets.duie import build_vocab


class CasRel(nn.Module):
    def __init__(self, bert_model_path, num_relations, bert_dim):
        super(CasRel, self).__init__()
        self.num_relations = num_relations
        self.bert_dim = bert_dim
        self.bert_model = BertModel.from_pretrained(bert_model_path)
        self.sub_heads_linear = nn.Linear(self.bert_dim, 1)
        self.sub_tails_linear = nn.Linear(self.bert_dim, 1)
        self.obj_heads_linear = nn.Linear(self.bert_dim, self.num_relations)
        self.obj_tails_linear = nn.Linear(self.bert_dim, self.num_relations)

    def forward(self, token_ids, token_type_ids, attention_mask, sub_head, sub_tail):
        encoded_text = \
            self.bert_model(input_ids=token_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)[
                0]  # (batch_size, seq, bert_dim )

        # subject预测
        pred_sub_heads = torch.sigmoid(self.sub_heads_linear(encoded_text))
        pred_sub_tails = torch.sigmoid(self.sub_tails_linear(encoded_text))

        # object预测
        sub_head_mapping = sub_head.unsqueeze(1)  # (batch_size, seq)->(batch_size, 1, seq)
        sub_tail_mapping = sub_tail.unsqueeze(1)  # (batch_size, seq)->(batch_size, 1, seq)
        sub_head = torch.matmul(sub_head_mapping, encoded_text)
        sub_tail = torch.matmul(sub_tail_mapping, encoded_text)
        encoded_text = (sub_head + sub_tail) / 2 + encoded_text
        pred_obj_heads = torch.sigmoid(self.obj_heads_linear(encoded_text))
        pred_obj_tails = torch.sigmoid(self.obj_tails_linear(encoded_text))

        return pred_sub_heads, pred_sub_tails, pred_obj_heads, pred_obj_tails


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    bert_model_path = '/tmp/chinese-roberta-wwm-ext'
    texts = ['原告：张三', '被告：李四伍']
    sub_head = torch.tensor([[0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0]], dtype=torch.float)
    sub_tail = torch.tensor([[0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0]], dtype=torch.float)
    tokenizer = BertTokenizer.from_pretrained(bert_model_path)
    encoded_outputs = tokenizer(texts, return_tensors='pt', padding=True)
    token_ids, token_type_ids, attention_mask = encoded_outputs['input_ids'], encoded_outputs['token_type_ids'], \
                                                encoded_outputs['attention_mask']
    print(token_ids)
    print(token_type_ids)
    print(attention_mask)
    model = CasRel(bert_model_path, num_relations=2, bert_dim=768)
    pred_sub_heads, pred_sub_tails, pred_obj_heads, pred_obj_tails = model(token_ids, token_type_ids, attention_mask,
                                                                           sub_head, sub_tail)

    vocab, _ = build_vocab(os.path.join(bert_model_path, 'vocab.txt'))
