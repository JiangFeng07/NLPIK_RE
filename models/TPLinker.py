#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @Time  : 2022/11/1 10:57
# @Author: lionel
import torch
from torch import nn


class HandshakingKerner(nn.Module):
    def __init__(self, hidden_size, shaking_type):
        super(HandshakingKerner, self).__init__()
        self.shaking_type = shaking_type
        if shaking_type == 'cat':
            self.combine_fc = nn.Linear(2 * hidden_size * 2, hidden_size)

    def forward(self, seq_hiddens):
        batch_size, seq_len, hidden_size = seq_hiddens.size()
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
    def __init__(self, encoder, hidden_size=768):
        super(TPLinkerBert, self).__init__()
        self.encoder = encoder
        self.hidden_size = hidden_size


if __name__ == '__main__':
    a = torch.randn((5, 2, 3))
    b = torch.randn((5, 2, 3))
    c = torch.cat([b, a], dim=-1)
    print(c.size())
