import pandas as pd
import numpy as np
import math
import tqdm
import csv
import os
from tqdm import tqdm
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
import random
import igraph as ig
from scipy import sparse
import torch.nn.functional as F
from torch.autograd import Variable

from pathformer import Pathormer
from util import *

device = torch.device("cuda:{}".format(1))

biogrid_intact = load_biogrid_df('BIOGRIDALL','./data')
mayo = pd.read_csv('./data/Mayo_Exp_1a.csv',sep=' ')
RosMap = pd.read_csv('./data/RosMap_Exp_1a.csv',sep=' ')

data_ros, y_ros = ad_preprocess(RosMap)
adj_ros = adj_generate(data_ros,biogrid_intact,K=3000, reverse = False)
X_train, Y_train, X_test, Y_test = dataset_prepare(data_ros, y_ros)

p = 0.8
X_train, Y_train, X_test, Y_test = dataset_prepare(X, Y, p)
num_sample = X.shape[0]
res_dir = './rosmap_results'
pathformer = Pathormer(n_copy=6, copy_dim=20, d_model=8, n_head=2, d_ff=16, drop_out=0.25, n_layers=2, hidden_dim=128,
                 out_hidden=128, Nnodes=3000, out_dim=2, device=None)
loss_func =nn.CrossEntropyLoss()
optimizer = optim.Adam(pathformer.parameters(),lr=0.001)
pathformer.to(device)

# Train the model
EPOCH = 20
batch_size = 8
#pbar = tqdm(range(EPOCH))
pathformer.train()
for i in range(EPOCH):
    # N = int(num_sample * 0.8)
    N = X_train.size(0)
    acc_list = []
    idx = list(range(N))
    random.shuffle(idx)
    X_epoch = X_train[idx]
    Y_epoch = Y_train[idx]

    acc_list = []
    loss_list = []
    total_B = 0
    pbar = tqdm(range(N // batch_size + 1))
    #for k in range(N // batch_size + 1):
    for k in pbar:
        if (k + 1) * batch_size <= N:
            batch_x = X_epoch[k * batch_size:(k + 1) * batch_size]
            batch_y = Y_epoch[k * batch_size:(k + 1) * batch_size]
            # idx = list(range(batch_size))
            B = batch_size
        else:
            batch_x = X_epoch[k * batch_size:]
            batch_y = Y_epoch[k * batch_size:]
            # N = int(num_sample * 0.8)
            # idx = list(range(N - (N // batch_size) * batch_size))
            B = N - (N // batch_size) * batch_size
        # random.shuffle(idx)
        # batch_x = batch_x[idx]
        # batch_y = batch_y[idx]
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        A = torch.stack([torch.FloatTensor(adj_ros)] * B, dim=0)
        A = A.to(device)
        pred, _ = pathformer(batch_x.unsqueeze(2), A)
        loss = loss_func(pred, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        target_true = torch.sum(torch.argmax(pred, dim=-1) == batch_y).item()
        acc = target_true / B
        acc_list.append(target_true)
        total_B += B
        loss_list.append(loss.item() * B)
        pbar.set_description('Epoch: {0} loss:{1:.3f} acc: {2:.4f}, :num: {3}'.format(i, loss.item(), acc, target_true))

    if epoch % 50 == 0:
        print("save current model...")
        model_name = os.path.join(res_dir, 'model_checkpoint{}.pth'.format(i))
        optimizer_name = os.path.join(res_dir, 'optimizer_checkpoint{}.pth'.format(i))
        #scheduler_name = os.path.join(args.res_dir, 'scheduler_checkpoint{}.pth'.format(epoch))
        torch.save(pathformer.state_dict(), model_name)
        torch.save(optimizer.state_dict(), optimizer_name)
        #torch.save(scheduler.state_dict(), scheduler_name)

    print('Epoch:', i ,'Train accuracy:', np.sum(acc_list)/total_B, 'train loss', np.sum(loss_list)/total_B)



N = X_test.size(0)
acc_list = []
loss_list = []
for k in range(N // batch_size + 1):
    if (k + 1) * batch_size <= N:
        batch_x = X_test[k * batch_size:(k + 1) * batch_size]
        batch_y = Y_test[k * batch_size:(k + 1) * batch_size]
        # idx = list(range(batch_size))
        B = batch_size
    else:
        batch_x = X_test[k * batch_size:]
        batch_y = Y_test[k * batch_size:]
        # N = int(num_sample * 0.8)
        # idx = list(range(N - (N // batch_size) * batch_size))
        B = N - (N // batch_size) * batch_size
    # random.shuffle(idx)
    # batch_x = batch_x[idx]
    # batch_y = batch_y[idx]
    batch_x = batch_x.to(device)
    batch_y = batch_y.to(device)
    A = torch.stack([torch.FloatTensor(adj)] * B, dim=0)
    A = A.to(device)
    with torch.no_grad():
        pred, _ = pathformer(batch_x.unsqueeze(2), A)
        loss = loss_func(pred, batch_y)

    target_true = torch.sum(torch.argmax(pred, dim=-1) == batch_y).item()
    acc_list.append(target_true)
    total_B += B
    loss_list.append(loss.item() * B)
    #pbar.set_description('Epoch: {0} loss:{1:.3f} acc: {2:.4f}, :num: {3}'.format(i, loss.item(), acc, target_true))

print('Test accuracy:', np.sum(acc_list)/N,'test loss', np.sum(loss_list)/N)