#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @Time  : 2022/11/1 11:37
# @Author: lionel
import json

import torch
from torch.utils import data
from transformers import BertTokenizer


class DytDataset(data.Dataset):
    def __init__(self, rel2id, file_path):
        self.rel2id = rel2id
        self.id2rel = {val: key for key, val in rel2id.items()}

        self.tag2id_ent = {'O': 0, 'EH_ET': 1}
        self.id2tag_ent = {val: key for key, val in self.tag2id_ent.items()}

        self.tag2id_head_rel = {'O': 0, 'SH_OH': 1, 'OH_SH': 2}
        self.id2tag_head_rel = {val: key for key, val in self.tag2id_head_rel.items()}

        self.tag2id_tail_rel = {'O': 0, 'ST_OT': 1, 'OT_ST': 2}
        self.id2tag_tail_rel = {val: key for key, val in self.tag2id_tail_rel.items()}

        self.data = json.load(open(file_path, 'r'))

    def __getitem__(self, index):
        text = self.data[index]['text']
        eh_et, sh_oh, oh_sh, st_ot, ot_st = set(), [], [], [], []
        for rel in self.data[index]['relation_list']:
            predicate = self.rel2id[rel['predicate']]
            sub_token_start, sub_token_end = rel['subj_tok_span']
            obj_token_start, obj_token_end = rel['obj_tok_span']

            eh_et.add((sub_token_start, sub_token_end - 1))
            eh_et.add((obj_token_start, obj_token_end - 1))
            if sub_token_start <= obj_token_start:
                sh_oh.append((predicate, sub_token_start, obj_token_start, self.tag2id_head_rel['SH_OH']))
            else:
                oh_sh.append((predicate, obj_token_start, sub_token_start, self.tag2id_head_rel['OH_SH']))

            if sub_token_end <= obj_token_end:
                st_ot.append((predicate, sub_token_end - 1, obj_token_end - 1, self.tag2id_tail_rel['ST_OT']))
            else:
                ot_st.append((predicate, obj_token_end - 1, sub_token_end - 1, self.tag2id_tail_rel['OT_ST']))

        return text, self.data[index]['relation_list'], eh_et, sh_oh, oh_sh, st_ot, ot_st

    def __len__(self):
        return len(self.data)


def collate_fn(batch, tokenizer, tag2id_ent, rel2id):
    texts, rels, eh_ets, sh_ohs, oh_shs, st_ots, ot_sts = zip(*batch)
    encode_outputs = tokenizer.batch_encode_plus(texts, return_tensors='pt', padding=True, add_special_tokens=False)
    token_ids, token_type_ids, attention_mask = tuple(encode_outputs.values())
    batch_size, max_seq = token_ids.size()

    shaking_head_tail_matrix_index = {}
    for i in range(max_seq):
        for j in range(i, max_seq):
            shaking_head_tail_matrix_index['%d_%d' % (i, j)] = len(shaking_head_tail_matrix_index)
    shaking_seq_len = max_seq * (max_seq + 1) // 2
    entities = torch.zeros((batch_size, shaking_seq_len)).long()
    for index, eh_et in enumerate(eh_ets):
        for eh, et in eh_et:
            entities[index][shaking_head_tail_matrix_index['%d_%d' % (eh, et)]] = tag2id_ent['EH_ET']

    subj_head_obj_head_matrix = torch.zeros((batch_size, len(rel2id), shaking_seq_len))
    for index, sh_oh in enumerate(sh_ohs):
        for predicate, sh, oh, tag in sh_oh:
            subj_head_obj_head_matrix[index][predicate][shaking_head_tail_matrix_index['%d_%d' % (sh, oh)]] = tag
    for index, oh_sh in enumerate(oh_shs):
        for predicate, oh, sh, tag in oh_sh:
            subj_head_obj_head_matrix[index][predicate][shaking_head_tail_matrix_index['%d_%d' % (oh, sh)]] = tag

    subj_tail_obj_tail_matrix = torch.zeros((batch_size, len(rel2id), shaking_seq_len))
    for index, st_ot in enumerate(st_ots):
        for predicate, st, ot, tag in st_ot:
            subj_tail_obj_tail_matrix[index][predicate][shaking_head_tail_matrix_index['%d_%d' % (st, ot)]] = tag
    for index, ot_st in enumerate(ot_sts):
        for predicate, ot, st, tag in ot_st:
            subj_tail_obj_tail_matrix[index][predicate][shaking_head_tail_matrix_index['%d_%d' % (ot, st)]] = tag

    return entities, subj_head_obj_head_matrix, subj_tail_obj_tail_matrix


if __name__ == '__main__':
    rel2id = json.load(open('../data/nyt/rel2id.json', 'r'))
    bert_model_path = '/tmp/bert_base_cased'
    tokenizer = BertTokenizer.from_pretrained(bert_model_path)
    dataset = DytDataset(rel2id, '../data/nyt/valid_data.json')

    dataloader = data.DataLoader(dataset, batch_size=10, shuffle=False,
                                 collate_fn=lambda batch: collate_fn(batch, tokenizer, tag2id_ent={'O': 0, 'EH_ET': 1},
                                                                     rel2id=rel2id))

    for ele in dataloader:
        print(ele[1])
        break
