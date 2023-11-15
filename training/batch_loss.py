import torch
from torch import nn
import math
import numpy as np
def max_drawdown(pc_array):
    """calculate the max drawdown with the portfolio changes
    @:param pc_array: all the portfolio changes during a trading process
    @:return: max drawdown
    """
    portfolio_values = []
    drawdown_list = []
    max_benefit = 0
    for i in range(pc_array.shape[0]):
        if i > 0:
            portfolio_values.append(portfolio_values[i - 1] * pc_array[i])
        else:
            portfolio_values.append(pc_array[i])
        if portfolio_values[i] > max_benefit:
            max_benefit = portfolio_values[i]
            drawdown_list.append(0.0)
        else:
            drawdown_list.append(1.0 - portfolio_values[i] / max_benefit)
    return max(drawdown_list)
count=0
v=0
tst_pc_array1=torch.randn(0).cuda()
class Batch_Loss(nn.Module):
    def __init__(self, commission_ratio, interest_rate, gamma=0.1, beta=0.1, device="cuda"):
        super(Batch_Loss, self).__init__()
        self.gamma = gamma
        self.beta = beta
        self.commission_ratio = commission_ratio
        self.interest_rate = interest_rate
        self.device = device

    def forward(self, w, y):

        close_price = y
        w = w.float()
        global tst_pc_array1
        close_price = close_price.float()
        reward = torch.matmul(w, close_price)

        close_price = close_price.view(close_price.size()[0], close_price.size()[2], close_price.size()[1])
        ###############################################################################################################
        future_omega = w * close_price / reward
        wt = future_omega[:-1]
        wt1 = w[1:]
        pure_pc = 1 - torch.sum(torch.abs(wt - wt1), -1) * self.commission_ratio
        pure_pc = pure_pc.to(self.device)
        pure_pc = torch.cat([torch.ones([1, 1]).to(self.device), pure_pc], 0)
        pure_pc = pure_pc.view(pure_pc.size()[0], 1, pure_pc.size()[1])

        reward = reward * pure_pc
        reward = reward
        tst_pc_array = reward
        tst_pc_array1=torch.cat([tst_pc_array1, reward], 0)
        sr_reward = tst_pc_array1 - 1
        SR = (sr_reward.mean()-2/252) / sr_reward.std()*math.sqrt(252)
        MDD = max_drawdown(tst_pc_array)
        portfolio_value = torch.prod(reward, 0)

        batch_loss = -torch.log(reward)

        loss = batch_loss.mean()
        return loss, portfolio_value, SR, MDD
