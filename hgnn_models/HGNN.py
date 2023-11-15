from torch import nn
from hgnn_models import HGNN_conv
import torch.nn.functional as F


class HGNN(nn.Module):
    def __init__(self, in_ch, n_hid, dropout=0.5):
        super(HGNN, self).__init__()
        self.dropout = dropout
        self.hgc1 = HGNN_conv(in_ch, n_hid)


    def forward(self, x, G):
        x = F.relu(self.hgc1(x, G))
        x = F.dropout(x, self.dropout)

        return x
