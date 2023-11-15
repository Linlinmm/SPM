import torch
from torch import nn
import random

class InfoNCE_Loss(nn.Module):
    def __init__(self, device="cuda"):
        super(InfoNCE_Loss, self).__init__()

    def forward(self, HGAT_output, adj):
        '''infoNCE_loss'''
        S = 743
        temp = 0.5
        k = 1
        batch = HGAT_output.size(0)
        stock = HGAT_output.size(1)
        dim =HGAT_output.size(2)
        info_loss = 0

        for i in range(104):

            diff = torch.nonzero(adj[:,i] == 0, as_tuple=False)
            diff = torch.squeeze(diff)
            S = diff.shape[0]
            diff = diff.cpu().numpy()
            diff = diff.tolist()
            index = torch.LongTensor(random.sample(diff, S)).cuda()
            index_b = torch.index_select(HGAT_output,1, index)
            out_i = HGAT_output[:, i, :]
            out_i = torch.unsqueeze(out_i, 1)

            out_i_k = HGAT_output[:, i, :]#co1vd
            out_i_k = torch.unsqueeze(out_i_k, 1)

            positive = torch.exp(torch.matmul(out_i, out_i_k.transpose(2, 1).contiguous()) / temp).squeeze()
            negative = torch.sum(torch.exp(torch.matmul(out_i, index_b.transpose(2, 1).contiguous()) / temp), dim=-1).squeeze()
            nce_loss = -torch.log(positive / negative)
            nce_loss = torch.sum(nce_loss)

            zero =  torch.randn(0)

            if torch.isfinite(nce_loss)==True:
                info_loss += nce_loss
            else:
                info_loss += zero
        info_loss = info_loss / stock / batch

        return info_loss
class InfoNCE_Loss1(nn.Module):
    def __init__(self, device="cuda"):
        super(InfoNCE_Loss1, self).__init__()

    def forward(self, HGAT_output, adj):
        '''infoNCE_loss'''
        S = 743
        temp = 0.5
        k = 1

        batch = HGAT_output.size(0)
        stock = HGAT_output.size(1)
        dim =HGAT_output.size(2)
        info_loss = 0

        for i in range(61):

            diff = torch.nonzero(adj[:,i] == 0, as_tuple=False)
            diff = torch.squeeze(diff)

            S = diff.shape[0]
            diff = diff.cpu().numpy()
            diff = diff.tolist()
            index = torch.LongTensor(random.sample(diff, S)).cuda()

            index_b = torch.index_select(HGAT_output,1, index)

            out_i = HGAT_output[:, i, :]
            out_i = torch.unsqueeze(out_i, 1)

            out_i_k = HGAT_output[:, i, :]#cov1d
            out_i_k = torch.unsqueeze(out_i_k, 1)

            positive = torch.exp(torch.matmul(out_i, out_i_k.transpose(2, 1).contiguous()) / temp).squeeze()

            negative = torch.sum(torch.exp(torch.matmul(out_i, index_b.transpose(2, 1).contiguous()) / temp), dim=-1).squeeze()
            nce_loss = -torch.log(positive / negative)
            nce_loss = torch.sum(nce_loss)

            zero =  torch.randn(0)

            if torch.isfinite(nce_loss)==True:
                info_loss += nce_loss
            else:
                info_loss += zero
        info_loss = info_loss / stock / batch

        return info_loss
