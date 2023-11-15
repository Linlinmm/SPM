import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import Attn_head_adj,Attn_head



class HGAT(nn.Module):
    def __init__(self, nfeat, nhid,  dropout):
        ''' attention mechanism. '''
        super(HGAT, self).__init__()
        self.dropout = dropout

        self.attention1 = Attn_head_adj(nfeat, nhid, in_drop=0.6, coef_drop=0.6, activation=nn.ELU(),
                                         residual=True)
        self.attention2 = Attn_head(nfeat, nhid, in_drop=0.5, coef_drop=0.5, activation=nn.ELU(),
                                        residual=True)

    def forward(self, x, G2,G1):


        x1 = self.attention1(x, G1)
        x2=self.attention2(x, G2)
        info_loss = info_loss_func(x1, adj)
        info_loss1 = info_loss_func1(x2, H)
        x=x1+x2
        x = F.dropout(x, self.dropout, training=self.training)

        return x,info_loss,info_loss1
