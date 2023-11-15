import torch
import torch.nn.functional as F
import time
from sklearn import metrics
import torch.utils.data as Data
from load_data import load_EOD_data, get_batch, get_industry_adj,get_fund_adj_H
from batch_loss_test import Batch_Loss
import pandas as pd
import numpy as np
import os
import csv
import codecs
from stofund25.hgnn_models.hypergraph_utils import generate_G_from_H
def data_write_csv(file_name, datas):
    file_csv = codecs.open(file_name, 'w+', 'utf-8')
    writer = csv.writer(file_csv, delimiter=' ', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
    for data in datas:
        writer.writerow(data)

fund_adj_tensor_H = get_fund_adj_H()
adj = torch.Tensor(get_industry_adj())
eod_data, ground_truth = load_EOD_data()

fund_adj_tensor_H = torch.Tensor(fund_adj_tensor_H)#【28，758，62】

Htensor2 = torch.randn(0)


for i in range(28):
    Htensor = torch.randn(0)
    for j in range(61):
        Htensor = torch.cat([Htensor, torch.Tensor(fund_adj_tensor_H[i]).unsqueeze(0)], dim=0)

    Htensor2 = torch.cat([Htensor2, torch.Tensor(Htensor)], dim=0)
loss_func = Batch_Loss(0.0025, 0.001, 0.01, 0.01, device="cuda")

def eval_epoch(model, validation_data, device, args):
    ''' Epoch operation in evaluation phase '''

    model.eval()
    i = 0

    total_loss = 0
    SR1=0
    total_portfolio_value = 1
    batch_size = args.batch_size
    stock = adj.shape[0]
    out = torch.ones(batch_size, 1, stock) / stock



    with torch.no_grad():
        for step, (eod, gt) in enumerate(validation_data):
            H = Htensor2[args.batch_size * step]
            Eod, Gt = eod.to(device), gt.to(device)
            G1 = generate_G_from_H(adj)

            G1 = torch.Tensor(G1)
            G1 = G1.to(device)

            G2 = generate_G_from_H(H)

            G2 = torch.Tensor(G2)
            G2 = G2.to(device)

            out = model(Eod, G2, G1,out, args.hidden)
            loss, portfolio_value, SR, MDD, lossData,volatility = loss_func(out, Gt, total_portfolio_value)
            if i == 0:
                ld = lossData
            else:

                ld = np.hstack([ld, lossData])

            i = i + 1

            total_loss += loss.item()
            total_portfolio_value *= portfolio_value.item()
            t1 = volatility
            SR1 = SR.item()

        return total_loss, total_portfolio_value, SR, MDD, ld,t1


list=[[]]
def train(model, test_data, optimizer, device, args):

    log_dir = args.save_model + '.chkpt'
    if os.path.exists(log_dir):
        checkpoint = torch.load(log_dir)
        model.load_state_dict(checkpoint['model'])

        start_epoch = checkpoint['epoch']
        print('load epoch {} successfully！'.format(start_epoch))

    start = time.time()
    valid_loss, valid_total_portfolio, valid_SR, valid_MDD, ld, volatility = eval_epoch(model, test_data, device,
                                                                args=args)
    print(valid_total_portfolio, valid_SR, valid_MDD,volatility)


def prepare_dataloaders(eod_data, gt_data, args):
    # ========= Preparing DataLoader =========#
    EOD, GT = [], []
    for i in range(eod_data.shape[1] - args.length):
        eod, gt = get_batch(eod_data, gt_data, i, args.length)
        EOD.append(eod)
        GT.append(gt)

    test_eod, test_gt = EOD[args.valid_index:], GT[args.valid_index:]

    test_eod =  torch.FloatTensor(test_eod)
    test_gt = torch.FloatTensor(test_gt)
    test_gt = test_gt.unsqueeze(2)
    test_dataset = Data.TensorDataset(test_eod, test_gt)
    test_loader = Data.DataLoader(dataset=test_dataset, batch_size=args.batch_size, drop_last=True)

    return test_loader
