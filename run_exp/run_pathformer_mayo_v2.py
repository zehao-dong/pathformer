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
import argparse
import pickle
from util import *
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

import matplotlib.pyplot as plt
from models import *
from pathformer import topK_Pathormer

# PARSE ARGUMENTS FROM COMMAND LINE
def arg_parse():
    parser = argparse.ArgumentParser(description='Argparser for CapsGNN')
    # ADD FOLLOWING ARGUMENTS
    parser.add_argument('--cuda', dest = 'cuda',
                help = 'CUDA.')
    parser.add_argument('--mode', dest = 'mode',
                help = 'cpu or gpu')
    parser.add_argument('--dropout', type=float, default=0.25)
    parser.add_argument('--n_copy', type=int, default=6, help='6: catogories of copy number')
    parser.add_argument('--copy_dim', type=int, default=20, help='20: dimension of catogory')
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--d_model', type=int, default=8)
    parser.add_argument('--n_head', type=int, default=2)
    parser.add_argument('--d_ff', type=int, default=16)
    parser.add_argument('--n_nodes', type=int, default=3000, help='3000: preprocessing 3000 genes according to variance')
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--out_hidden', type=int, default=128)
    parser.add_argument('--out_dim', type=int, default=2)
    parser.add_argument('--seed', type=int, default=0)#np.random.randint(1,high=10000,size=1)[0]
    parser.add_argument('--device', type=bool, default=None)
    parser.add_argument('--K', type=int, default=1000,help='1000: selected top K genes')
    parser.add_argument('--lr', dest = 'learning_rate', type = float, default=0.001,
                help = '0.001: Learning rate.')
    parser.add_argument('--batch-size', dest = 'batch_size', type = int, default=8,
                help = 'Batch size.')
    parser.add_argument('--epochs', dest = 'num_epochs', type = int, default=20,
                help = 'Number of epochs to train.')
    parser.add_argument('--dataset', type = str, default="mayo",
                help = 'which dataset and same genes or not.')  
    
    

    # SET DEFAULT INPUT ARGUMENT
    parser.set_defaults(cuda = 0,
                        mode = 'gpu')
    #parser.extract_gcn_dims= [int(x) for x in parser.extract_gcn_dims.split('-')]
    #parser.extract_nodes= [int(x) for x in parser.extract_nodes.split('-')]
    ##parser.gcn_dims= [int(x) for x in parser.gcn_dims.split('-')]
    #parser.cluster_nums= [int(x) for x in parser.cluster_nums.split('-')]
    return parser.parse_args()

if __name__ == '__main__':
    args =  arg_parse()
    print(args)
    print(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    if args.mode == 'gpu':
        device = torch.device('cuda:{}'.format(args.cuda))
    else:
        device = torch.device("cpu")
    # device = torch.device("cuda:{}".format(0))

    torch.manual_seed(args.seed)
    # np.random.seed(args.seed)
    # random.seed(args.seed)

    biogrid_intact = load_biogrid_df('BIOGRIDALL','./data')


    
    if args.dataset == "mayo":
        mayo = pd.read_csv('./data/Mayo_Exp_1a.csv',sep=' ')
        data_mayo, y_mayo, data_no_var = ad_preprocess(mayo,args.n_nodes)
        gene_list = list(data_mayo.index) #show top 3000 gene names based on variance

    elif args.dataset == "rosmap":
        rosmap = pd.read_csv('./data/RosMap_Exp_1a.csv',sep=' ')
        data_mayo, y_mayo, data_no_var = ad_preprocess(rosmap,args.n_nodes)
        gene_list = list(data_mayo.index) #show top 3000 gene names based on variance

    elif args.dataset == "same":
        mayo = pd.read_csv('./data/Mayo_Exp_1a.csv',sep=' ')
        mayo1 = pd.read_csv('./data/RosMap_Exp_1a.csv',sep=' ')
        data_mayo, y_mayo, data_no_var = ad_preprocess(mayo,args.n_nodes)
        gene_list = list(data_mayo.index) #show top 3000 gene names based on variance
        new_df = get_samegene(gene_list, mayo1)
        print("got 3000 same genes")
        data_mayo, y_mayo, data_no_var = ad_preprocess(new_df,args.n_nodes)
        gene_list2 = list(data_mayo.index)
 
    adj_mayo = adj_generate(data_mayo,biogrid_intact,args.n_nodes, reverse = False)#make adj matrix 3000*3000

    # X_train, Y_train, X_test, Y_test = dataset_prepare(data_mayo, y_mayo)

    p = 0.8
    X_train, Y_train, X_test, Y_test = dataset_prepare(data_mayo, y_mayo, p)

    num_sample = data_mayo.shape[0]#3000
    res_dir = './mayo_results'

    pathformer = topK_Pathormer(n_copy=args.n_copy, copy_dim=args.copy_dim, d_model=args.d_model, n_head=args.n_head, d_ff=args.d_ff, drop_out=args.dropout, n_layers=args.n_layers, hidden_dim=args.hidden_dim,
                    out_hidden=args.out_hidden, Nnodes=args.n_nodes, out_dim=args.out_dim, device=args.device,K=args.K)
    


    def loss_func(logits, labels, fold_change, lambda_reg):
        # Calculate the cross-entropy loss
        ce_loss = F.cross_entropy(logits, labels)
        
        fc_loss = lambda_reg * torch.mean(fold_change**2)

        total_loss = ce_loss + fc_loss
        
        return total_loss
    # loss_func =nn.CrossEntropyLoss()
    optimizer = optim.Adam(pathformer.parameters(),lr=args.learning_rate)
    pathformer.to(device)

    # Train the model
    EPOCH = args.num_epochs
    batch_size = args.batch_size
    #pbar = tqdm(range(EPOCH))
    pathformer.train()
    # graphnn.train()
    #     
    lambda_reg = 1  # Regularization parameter

    per_list1 = []
    accuracy_list1 = []
    rec_list1 = []
    f1_list1 = []

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
        pbar = tqdm.tqdm(range(N // batch_size + 1))
        #for k in range(N // batch_size + 1):
        fin_nppred = []
        fin_npbatch_y = []
        # print(pbar)
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
            # batch_x = batch_x[idx]
            # batch_y = batch_y[idx]
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            #8*3000*3000
            A = torch.stack([torch.FloatTensor(adj_mayo)] * B, dim=0)
            A = A.to(device)

            pred, indexes, attn_list = pathformer(batch_x.unsqueeze(2), A)

            nppred1 = (pred.cpu().detach().numpy()).tolist()
            nppred1 = [p.index(max(p)) for p in nppred1]
            fin_nppred.extend(nppred1)

            npbatch_y1 = (batch_y.cpu().detach().numpy()).tolist()
            fin_npbatch_y.extend(npbatch_y1)


            indexes_cpu = indexes.cpu().numpy()
            selected_data = data_no_var.iloc[indexes_cpu]
            fold_change_1 = log2fc(selected_data, y_mayo)
            fold_change_tensor = torch.from_numpy(fold_change_1).to(device)


            loss = loss_func(pred, batch_y, fold_change_tensor, lambda_reg)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            target_true = torch.sum(torch.argmax(pred, dim=-1) == batch_y).item()
            acc = target_true / B
            acc_list.append(target_true)
            total_B += B
    
            loss_list.append(loss.item() * B)

            pbar.set_description('Epoch: {0} loss:{1:.3f} acc: {2:.4f}, :num: {3}'.format(i, loss.item(), acc, target_true))
        

        if total_B % 50 == 0: # Checks whether the total number of processed samples is divisible by 50
            print("save current model...")
            # Creates a file path for saving the current model checkpoint, including the current epoch number in the filename
            model_name = os.path.join(res_dir, 'model_checkpoint{}.pth'.format(i))
            # Creates a file path for saving the current optimizer checkpoint, including the current epoch number in the filename.
            optimizer_name = os.path.join(res_dir, 'optimizer_checkpoint{}.pth'.format(i))
            
            #scheduler_name = os.path.join(args.res_dir, 'scheduler_checkpoint{}.pth'.format(epoch))
            
            # Saves the current state of the model to a file
            torch.save(pathformer.state_dict(), model_name)
            # Saves the current state of the optimizer to a file
            torch.save(optimizer.state_dict(), optimizer_name)
            #torch.save(scheduler.state_dict(), scheduler_name)

        
            
        per_score = precision_score(fin_npbatch_y, fin_nppred, average='weighted')
        acc_score = accuracy_score(fin_npbatch_y, fin_nppred)
        rec_score = recall_score(fin_npbatch_y, fin_nppred, average='weighted')
        f_score = f1_score(fin_npbatch_y, fin_nppred, average='weighted')

        per_list1.append(per_score)
        accuracy_list1.append(acc_score)
        rec_list1.append(rec_score)
        f1_list1.append(f_score)

        

        print('Epoch:', i ,'Train accuracy:', np.sum(acc_list)/total_B, 'train loss', np.sum(loss_list)/total_B)
        print('f1_score:', f_score, 'percision', per_score, 'recall', rec_score)
    
    
    #画图用
    # names = [str(x) for x in range(EPOCH)]
    # x_axis = np.arange(len(names))

    # plt.bar(x_axis -0.3,per_list1,width=0.2,label = 'percision')
    # plt.bar(x_axis -0.1,accuracy_list1,width=0.2,label = 'accuracy')
    # plt.bar(x_axis +0.1,rec_list1,width=0.2,label = 'recall')
    # plt.bar(x_axis +0.3,f1_list1,width=0.2,label = 'f1score')

    # plt.xticks(x_axis, names)
    # plt.legend()

    # plt.title('different statistical scores in training')
    # plt.xlabel('training_epoch')
    # plt.ylabel('score')
    # plt.show()


    N = X_test.size(0)

    batch_size = 7
    acc_list = []
    loss_list = []

    per_list = []
    accuracy_list = []
    rec_list = []
    f1_list = []

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

        A = torch.stack([torch.FloatTensor(adj_mayo)] * B, dim=0)
        A = A.to(device)
        with torch.no_grad():
            pred, indexes, attn_list = pathformer(batch_x.unsqueeze(2), A)

            indexes_cpu = indexes.cpu().numpy()
            selected_data = data_no_var.iloc[indexes_cpu]
            fold_change = log2fc(selected_data, y_mayo)
            fold_change_tensor = torch.from_numpy(fold_change).to(device)

            loss = loss_func(pred, batch_y, fold_change_tensor, lambda_reg)

        nppred = (pred.cpu().detach().numpy()).tolist()
        nppred = [p.index(max(p)) for p in nppred]
        

        npbatch_y = (batch_y.cpu().detach().numpy()).tolist()
        

        target_true = torch.sum(torch.argmax(pred, dim=-1) == batch_y).item()
        acc_list.append(target_true)
        # total_B += B
        loss_list.append(loss.item() * B)
        # pbar.set_description('Epoch: {0} loss:{1:.3f} acc: {2:.4f}, :num: {3}'.format(i, loss.item(), acc, target_true))

    print('Test accuracy:', np.sum(acc_list)/N,'test loss', np.sum(loss_list)/N)


