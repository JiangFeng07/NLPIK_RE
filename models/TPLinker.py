#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @Time  : 2022/11/1 10:57
# @Author: lionel
import torch
from torch import nn
from transformers import BertTokenizer, BertModel


class HandshakingKerner(nn.Module):
    def __init__(self, hidden_size, shaking_type):
        super(HandshakingKerner, self).__init__()
        self.shaking_type = shaking_type
        if shaking_type == 'cat':
            self.combine_fc = nn.Linear(2 * hidden_size, hidden_size)

    def forward(self, seq_hiddens):
        """

        :param seq_hiddens:  (batch_size, seq_len, hidden_size)
        :return:
        """
        seq_len = seq_hiddens.size(1)
        shaking_hiddens_list = []
        for ind in range(seq_len):
            hidden_each_step = seq_hiddens[:, ind, :]
            repeat_hiddens = hidden_each_step.unsqueeze(1).repeat(1, seq_len - ind, 1)
            visible_hiddens = seq_hiddens[:, ind:, :]  # only look back
            if self.shaking_type == 'cat':
                shaking_hiddens = torch.cat([repeat_hiddens, visible_hiddens], dim=-1)
                shaking_hiddens = torch.tanh(self.combine_fc(shaking_hiddens))
                shaking_hiddens_list.append(shaking_hiddens)

        long_shaking_hiddens = torch.cat(shaking_hiddens_list, dim=1)
        return long_shaking_hiddens


class TPLinkerBert(nn.Module):
    def __init__(self, encoder, hidden_size, shaking_type, rel_size):
        super(TPLinkerBert, self).__init__()
        self.encoder = encoder
        self.hidden_size = hidden_size
        self.entity_fc = nn.Linear(hidden_size, 2)
        self.rel_head_fcs = [nn.Linear(hidden_size, 3) for _ in range(rel_size)]
        self.rel_tail_fcs = [nn.Linear(hidden_size, 3) for _ in range(rel_size)]
        self.handshaking_kerner = HandshakingKerner(hidden_size, shaking_type)

    def forward(self, token_ids, token_type_ids, attention_mask):
        encoder_outputs = self.encoder(input_ids=token_ids, token_type_ids=token_type_ids,
                                       attention_mask=attention_mask)[0]  # last layer output

        shaking_hiddens = self.handshaking_kerner(encoder_outputs)

        entity_outputs = self.entity_fc(shaking_hiddens)

        rel_head_outputs = []
        for fc in self.rel_head_fcs:
            rel_head_outputs.append(fc(shaking_hiddens))

        rel_tail_outputs = []
        for fc in self.rel_tail_fcs:
            rel_tail_outputs.append(fc(shaking_hiddens))

        rel_head_outputs = torch.stack(rel_head_outputs, dim=1)
        rel_tail_outputs = torch.stack(rel_tail_outputs, dim=1)

        return entity_outputs, rel_head_outputs, rel_tail_outputs


class TPLinkerLoss(nn.Module):
    def __init__(self, mask):
        super(TPLinkerLoss, self).__init__()
        self.mask = mask

    def forward(self, entities, pre_entities, rel_heads, pre_rel_heads, rel_tails, pre_rel_tails):
        entity_loss = nn.CrossEntropyLoss(pre_entities.view(-1, pre_entities.size(-1)), entities.view(-1))
        rel_heads_loss = nn.CrossEntropyLoss(rel_heads.view(-1, pre_rel_heads.size(-1)), rel_heads.view(-1))
        rel_tails_loss = nn.CrossEntropyLoss(rel_heads.view(-1, pre_rel_tails.size(-1)), rel_tails.view(-1))

        return entity_loss + rel_heads_loss + rel_tails_loss


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bert_model_path = '/tmp/chinese-roberta-wwm-ext'
    texts = ['原告：张三', '被告：李四伍']
    tokenizer = BertTokenizer.from_pretrained(bert_model_path)
    encode_outputs = tokenizer.batch_encode_plus(texts, return_tensors='pt', padding=True, add_special_tokens=False)
    token_ids, token_type_ids, attention_mask = tuple(encode_outputs.values())
    bert_model = BertModel.from_pretrained(bert_model_path)
    hidden_size = bert_model.config.hidden_size
    tplinker = TPLinkerBert(encoder=bert_model, hidden_size=hidden_size, shaking_type='cat', rel_size=2)
    entity_outputs, rel_head_outputs, rel_tail_outputs = tplinker(token_ids, token_type_ids, attention_mask)
    print(token_ids.size())
    print(entity_outputs.size())
    print(entity_outputs.view(-1, entity_outputs.size(-1)).size())
    print(rel_head_outputs.size())
    print(rel_tail_outputs.size())
