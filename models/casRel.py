#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @Time  : 2022/10/24 09:44
# @Author: lionel
import os

import torch
from torch import nn

from transformers import BertModel, BertTokenizer
import torch.nn.functional as F


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

    def get_subs(self, token_ids, token_type_ids, attention_mask):
        self.encoded_text = \
            self.bert_model(input_ids=token_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)[
                0]  # (batch_size, seq, bert_dim )

        # subject预测
        pred_sub_heads = torch.sigmoid(self.sub_heads_linear(self.encoded_text))
        pred_sub_tails = torch.sigmoid(self.sub_tails_linear(self.encoded_text))
        return pred_sub_heads, pred_sub_tails

    def forward(self, token_ids, token_type_ids, attention_mask, sub_head, sub_tail):
        # subject预测
        pred_sub_heads, pred_sub_tails = self.get_subs(token_ids, token_type_ids, attention_mask)

        # object预测
        sub_head_mapping = sub_head.unsqueeze(1)  # (batch_size, seq)->(batch_size, 1, seq)
        sub_tail_mapping = sub_tail.unsqueeze(1)  # (batch_size, seq)->(batch_size, 1, seq)
        sub_head = torch.matmul(sub_head_mapping, self.encoded_text)
        sub_tail = torch.matmul(sub_tail_mapping, self.encoded_text)
        encoded_text = (sub_head + sub_tail) / 2 + self.encoded_text
        pred_obj_heads = torch.sigmoid(self.obj_heads_linear(encoded_text))
        pred_obj_tails = torch.sigmoid(self.obj_tails_linear(encoded_text))

        return pred_sub_heads, pred_sub_tails, pred_obj_heads, pred_obj_tails


class MyLoss(nn.Module):
    def __init__(self, mask):
        super(MyLoss, self).__init__()
        self.mask = mask
        self.mask2 = mask.unsqueeze(-1)

    def loss_fn(self, predict_label, gold_label):
        """

        :param predict_label: 预测结果 （batch_size, seq, 1）
        :param gold_label: 真实标签（batch_size, seq）
        :param mask: (batch_size, seq)
        :return: bce loss
        """
        predict_label = predict_label.squeeze(-1)
        loss = F.binary_cross_entropy(predict_label, gold_label, reduction='none')
        if self.mask.shape == loss.shape:
            loss = torch.sum(self.mask * loss) / torch.sum(self.mask)
        else:
            loss = torch.sum(self.mask2 * loss) / torch.sum(self.mask2)
        return loss

    def forward(self, predict_label, gold_label):
        sub_heads_loss = self.loss_fn(predict_label['pred_sub_heads'], gold_label['sub_heads'])
        sub_tails_loss = self.loss_fn(predict_label['pred_sub_tails'], gold_label['sub_tails'])
        obj_heads_loss = self.loss_fn(predict_label['pred_obj_heads'], gold_label['obj_heads'])
        obj_tails_loss = self.loss_fn(predict_label['pred_obj_tails'], gold_label['obj_tails'])
        return sub_heads_loss + sub_tails_loss + obj_heads_loss + obj_tails_loss


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
    model = CasRel(bert_model_path, num_relations=48, bert_dim=768)
    pred_sub_heads, pred_sub_tails, pred_obj_heads, pred_obj_tails = model(token_ids, token_type_ids, attention_mask,
                                                                           sub_head, sub_tail)
    print(pred_obj_heads)

    # input = torch.randn((2, 2), requires_grad=True)
    # target = torch.rand((2, 2), requires_grad=False)
    # loss = F.binary_cross_entropy(F.sigmoid(input), target, reduction='none')
    # print(torch.sum(loss) / 4)
    # loss = F.binary_cross_entropy(F.sigmoid(input), target)
    # print(loss)
