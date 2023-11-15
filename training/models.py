import torch
import torch.nn as nn
import torch.nn.functional as F
from module import HGAT
import numpy as np


class HGAM(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(
            self,
            rnn_unit, n_hid,
            feature,
            dropout,
            ):

        super().__init__()

        # self.linear = nn.Linear(feature, d_word_vec)
        self.rnn = nn.GRU(feature,
                          rnn_unit,
                          num_layers=2,
                          batch_first=True,
                          bidirectional=False)

        self.HGAT = HGAT(nfeat=rnn_unit,
                         nhid=n_hid,
                         dropout=dropout,
                         )
        self.linear_out = nn.Linear(in_features=1 + n_hid, out_features=1)

        self.dropout = dropout



    def forward(self, src_seq,G2,G1, previous_w,  n_hid):

        batch = src_seq.size(0)
        stock = src_seq.size(1)
        seq_len = src_seq.size(2)
        dim = src_seq.size(3)
        src_seq = torch.reshape(src_seq, (batch*stock, seq_len, dim))
        rnn_output, *_ = self.rnn(src_seq)
        rnn_output = rnn_output[:, -1, :]
        enc_output = torch.reshape(rnn_output, (batch, stock, -1))

        HGAT_output,info_loss,info_loss1 = self.HGAT(enc_output,G2,G1)

        HGAT_output = torch.reshape(HGAT_output, (batch*stock, -1))

        HGAT_output = torch.reshape(HGAT_output, (batch, stock, -1))

        previous_w = previous_w.permute(0, 2, 1)
        out = torch.cat([HGAT_output.cuda(), previous_w.cuda()], 2)
        out = self.linear_out(out)
        out = out.permute(0, 2, 1)
        final_out = out
        final_out=F.softmax(final_out, dim=-1)

        return final_out,info_loss,info_loss1
