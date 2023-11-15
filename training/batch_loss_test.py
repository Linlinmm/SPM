import torch
from torch import nn
import math
import numpy as np
import csv
import codecs
import pandas as pd

def data_write_csv(file_name, datas):
    file_csv = codecs.open(file_name, 'w+', 'utf-8')
    writer = csv.writer(file_csv, delimiter=' ', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
    for data in datas:
        writer.writerow(data)



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
pr2=torch.randn(0).cuda()
t4=[]
class Batch_Loss(nn.Module):
    def __init__(self, commission_ratio, interest_rate, gamma=0.1, beta=0.1, device="cuda"):
        super(Batch_Loss, self).__init__()
        self.gamma = gamma
        self.beta = beta
        self.commission_ratio = commission_ratio
        self.interest_rate = interest_rate
        self.device = device

    def forward(self, w, y,pr):

        close_price = y
        w = w.float()

        lossData = [[]]
        global tst_pc_array1
        global pr2
        global t4
        pr2=torch.tensor([pr]).cuda()
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
        for i in range(32):
            if i ==0:
               pr = torch.mul(pr2,reward[i,:,:])
               pr1=pr.squeeze(dim=-1)

               lossData.append(pr1.cpu().numpy())
            else:

               pr = torch.mul(pr, reward[i, :, :])
               pr1 = pr.squeeze(dim=-1)
               lossData.append(pr1.cpu().numpy())

        data_write_csv("/parametres.csv", lossData)
        t4=np.append(t4,lossData[1:])
        t3=tst_pc_array1.cpu().numpy()
        t3 = np.array(t3).flatten()

        t4 = np.array(t4).flatten()


        log_rets = np.log(t4)
        log_rets3 = np.log(t3)
        log_rets=np.diff(log_rets)
        log_rets3 = np.diff(log_rets3)

        log_rets_std = np.std(log_rets)
        log_rets_std3 = np.std(log_rets3)
        volatility_per = log_rets_std / log_rets.mean() / np.sqrt(1 / 252)


        portfolio_value = torch.prod(reward, 0)


        batch_loss = -torch.log(reward)

        loss = batch_loss.mean()
        return loss, portfolio_value, SR, MDD,lossData,volatility_per
