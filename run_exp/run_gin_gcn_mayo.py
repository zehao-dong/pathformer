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
import copy
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
    parser.add_argument('--pool', type=str, default='add')
    parser.add_argument('--gnn', type=str, default='gin')
    parser.add_argument('--K', type=int, default=1000,help='50: selected top K genes')
    parser.add_argument('--lr', dest = 'learning_rate', type = float, default=0.001,
                help = '0.001: Learning rate.')
    parser.add_argument('--batch-size', dest = 'batch_size', type = int, default=1,
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
    # print(args)
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
        data_mayo, y_mayo = ad_preprocess(mayo,args.n_nodes)
        gene_list = list(data_mayo.index) #show top 3000 gene names based on variance
        mayo_adj = pd.read_csv('./data/mayo_A.txt',sep=',',header=None)#mayo_A or rosmap_A
    elif args.dataset == "rosmap":
        rosmap = pd.read_csv('./data/RosMap_Exp_1a.csv',sep=' ')
        data_mayo, y_mayo = ad_preprocess(rosmap,args.n_nodes)
        gene_list = list(data_mayo.index) #show top 3000 gene names based on variance
        mayo_adj = pd.read_csv('./data/rosmap_A.txt',sep=',',header=None)#mayo_A or rosmap_A
    elif args.dataset == "same":
        mayo = pd.read_csv('./data/Mayo_Exp_1a.csv',sep=' ')
        mayo1 = pd.read_csv('./data/RosMap_Exp_1a.csv',sep=' ')
        data_mayo, y_mayo = ad_preprocess(mayo,args.n_nodes)
        gene_list = list(data_mayo.index) #show top 3000 gene names based on variance
        new_df = get_samegene(gene_list, mayo1)
        print("got 3000 same genes")
        data_mayo, y_mayo = ad_preprocess(new_df,args.n_nodes)
        gene_list2 = list(data_mayo.index)
        mayo_adj = pd.read_csv('./data/rosmap_A.txt',sep=',',header=None)#mayo_A or rosmap_A

    

    #----2-----
    # new_df = get_samegene(gene_list, mayo1)
    # print("got 3000 same genes")
    # data_mayo, y_mayo = ad_preprocess(new_df,args.n_nodes)
    # gene_list2 = list(data_mayo.index)
    # mayo_adj = pd.read_csv('./data/mayo_A.txt',sep=',',header=None)#mayo_A or rosmap_A

    c1=mayo_adj[0].astype(int)
    c2=mayo_adj[1].astype(int)
    c1=torch.tensor(c1).type(torch.LongTensor)
    c2=torch.tensor(c2).type(torch.LongTensor)
    # c1=c1.view(-1,1)
    # c2=c2.view(-1,1)
    c5=torch.stack((c1,c2))

    c3 = copy.deepcopy(c5)
    p = 0.8
    X_train, Y_train, X_test, Y_test = dataset_prepare(data_mayo, y_mayo, p)
    num_sample = data_mayo.shape[0]#3000
    # res_dir = './mayo_results' #args.n_nodes*args.batch_size
    graphnn = gnn_model(input_dim=1, emb_dim=args.n_nodes*args.batch_size, num_layers=args.n_layers, out_dim=args.out_dim, pool=args.pool, gnn_type=args.gnn,device=args.device, K = args.K)
    loss_func =nn.CrossEntropyLoss()
    # optimizer = optim.Adam(pathformer.parameters(),lr=args.learning_rate)
    optimizer = optim.Adam(graphnn.parameters(),lr=args.learning_rate)
    # pathformer.to(device)
    graphnn.to(device)

    # Train the model
    EPOCH = args.num_epochs
    batch_size = args.batch_size
    #pbar = tqdm(range(EPOCH))
    # pathformer.train()
    graphnn.train()

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
        pbar = tqdm.tqdm(range(N // batch_size))
        #for k in range(N // batch_size + 1):
        fin_nppred = []
        fin_npbatch_y = []
        print(pbar)
        for k in pbar:
            batch_x = X_epoch[k]
            batch_y = Y_epoch[k]
            B = batch_size
            # if (k + 1) * batch_size < N:   
            #     batch_x = X_epoch[k * batch_size:(k + 1) * batch_size]
            #     batch_y = Y_epoch[k * batch_size:(k + 1) * batch_size]
            #     # idx = list(range(batch_size))
            #     B = batch_size
            # else:
            #     batch_x = X_epoch[k * batch_size:]
            #     batch_y = Y_epoch[k * batch_size:]
            #     # N = int(num_sample * 0.8)
            #     # idx = list(range(N - (N // batch_size) * batch_size))
            #     B = N - (N // batch_size) * batch_size
            # batch_x = batch_x[idx]
            # batch_y = batch_y[idx]
            batch_y = batch_y.clone().detach()
            batch_y = batch_y.unsqueeze(0)
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            # A = torch.stack([torch.FloatTensor(adj_mayo)] * B, dim=0)
            # A = A.to(device)
            
            # pred, indexes = pathformer(batch_x.unsqueeze(2), A)
            c4=copy.deepcopy(c3)

            c10=torch.zeros(args.n_nodes,dtype=torch.long)
            c11=copy.deepcopy(c10)
            #这里要处理edge——index和batch了
            for i in range(batch_size-1):

                c4=torch.add(c4,args.n_nodes)
                c3=torch.cat((c3,c4),dim=1) #edge_index好了

                c11=torch.add(c11,1)
                c10=torch.cat((c10,c11))#batch好了

            c3 = c3.to(device)
            c10 = c10.to(device)
            
            batch_x=batch_x.view(batch_size*args.n_nodes)
            
            batch_x = batch_x.unsqueeze(1)

            pred, indexes = graphnn(batch_x, c3, c10)

            nppred1 = (pred.cpu().detach().numpy()).tolist()
            nppred1 = [p.index(max(p)) for p in nppred1]
            fin_nppred.append(nppred1[0])

            npbatch_y1 = [(batch_y.cpu().detach().numpy()).tolist()]
            fin_npbatch_y.extend(npbatch_y1)

            
            
   
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
        
        
        
        

        # if total_B % 50 == 0:
        #     print("save current model...")
        #     model_name = os.path.join(res_dir, 'model_checkpoint{}.pth'.format(i))
        #     optimizer_name = os.path.join(res_dir, 'optimizer_checkpoint{}.pth'.format(i))
        #     #scheduler_name = os.path.join(args.res_dir, 'scheduler_checkpoint{}.pth'.format(epoch))
        #     # torch.save(pathformer.state_dict(), model_name)

        #     torch.save(graphnn.state_dict(), model_name)

        #     torch.save(optimizer.state_dict(), optimizer_name)
        #     #torch.save(scheduler.state_dict(), scheduler_name)

        
            
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
    # after training, add topk genes into a txt file
    textfile2 = open("./data/result/GCN/rosmap.txt", "w") #save idx of Top K genes 
    for idx in indexes:
        if args.dataset == "same":
            textfile2.write(str(gene_list2[idx]) + "\n")# if mayo, gene_list, if rosmap, gene_list2
        else:
            textfile2.write(str(gene_list[idx]) + "\n")# if mayo, gene_list, if rosmap, gene_list2
    textfile2.close()

    names = [str(x) for x in range(EPOCH)]
    x_axis = np.arange(len(names))

    plt.bar(x_axis -0.3,per_list1,width=0.2,label = 'percision')
    plt.bar(x_axis -0.1,accuracy_list1,width=0.2,label = 'accuracy')
    plt.bar(x_axis +0.1,rec_list1,width=0.2,label = 'recall')
    plt.bar(x_axis +0.3,f1_list1,width=0.2,label = 'f1score')

    plt.xticks(x_axis, names)
    plt.legend()

    plt.title('different statistical scores in training')
    plt.xlabel('training_epoch')
    plt.ylabel('score')
    plt.show()


    N = X_test.size(0)
    acc_list = []
    loss_list = []

    per_list = []
    accuracy_list = []
    rec_list = []
    f1_list = []
    for k in range(N // batch_size):
        batch_x = X_epoch[k]
        batch_y = Y_epoch[k]
        B = batch_size
        # if (k + 1) * batch_size <= N:
        #     batch_x = X_test[k * batch_size:(k + 1) * batch_size]
        #     batch_y = Y_test[k * batch_size:(k + 1) * batch_size]
        #     # idx = list(range(batch_size))
        #     B = batch_size
        # else:
        #     batch_x = X_test[k * batch_size:]
        #     batch_y = Y_test[k * batch_size:]
        #     # N = int(num_sample * 0.8)
        #     # idx = list(range(N - (N // batch_size) * batch_size))
        #     B = N - (N // batch_size) * batch_size
        # random.shuffle(idx)
        # batch_x = batch_x[idx]
        # batch_y = batch_y[idx]
        batch_y = batch_y.clone().detach()
        batch_y = batch_y.unsqueeze(0)
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        # A = torch.stack([torch.FloatTensor(adj_mayo)] * B, dim=0)
        # A = A.to(device)

        c4=copy.deepcopy(c5)
        c10=torch.zeros(args.n_nodes,dtype=torch.long)
        c11=copy.deepcopy(c10)
        #这里要处理edge——index和batch了
        for i in range(batch_size-1):

            c4=torch.add(c4,args.n_nodes)
            c5=torch.cat((c5,c4),dim=1) #edge_index好了

            c11=torch.add(c11,1)
            c10=torch.cat((c10,c11))#batch好了

        # new=torch.gt(c3,23999)
        # print(new.any())
        c5 = c5.to(device)
        c10 = c10.to(device)
        
        batch_x=batch_x.view(batch_size*args.n_nodes)
        
        batch_x = batch_x.unsqueeze(1)
        # print(batch_x.size())
        # print(c3.size())
        # print(c10.size())

        
        with torch.no_grad():
            # pred, indexes = pathformer(batch_x.unsqueeze(2), A)
            pred, indexes = graphnn(batch_x, c3, c10)
            # pred, indexes = graphnn(batch_x.unsqueeze(2), A)
            loss = loss_func(pred, batch_y)



        nppred = (pred.cpu().detach().numpy()).tolist()
        nppred = [p.index(max(p)) for p in nppred]
        

        npbatch_y = (batch_y.cpu().detach().numpy()).tolist()
        

        target_true = torch.sum(torch.argmax(pred, dim=-1) == batch_y).item()
        acc_list.append(target_true)
        total_B += B
        loss_list.append(loss.item() * B)
        #pbar.set_description('Epoch: {0} loss:{1:.3f} acc: {2:.4f}, :num: {3}'.format(i, loss.item(), acc, target_true))

        # per_score = precision_score(npbatch_y, nppred, average='binary')
        # acc_score = accuracy_score(npbatch_y, nppred)
        # rec_score = recall_score(npbatch_y, nppred, average='binary')
        # f_score = f1_score(npbatch_y, nppred, average='binary')
        # # roc = roc_curve(npbatch_y, nppred, pos_label=1)
        # per_list.append(per_score)
        # accuracy_list.append(acc_score)
        # rec_list.append(rec_score)
        # f1_list.append(f_score)


    #after testing, add topk genes into a txt file
    # textfile2 = open("./data/rosmap_topk_test.txt", "w")
    # for idx in indexes:
    #     textfile2.write(str(gene_list[idx]) + "\n")
    # textfile2.close()


    # print("per_score are")
    # print(per_list)
    # print("acc_score are")
    # print(accuracy_list)
    # print("rec_score are")
    # print(rec_list)
    # print("f_score are")
    # print(f1_list)
    # names = [str(x) for x in range(N // batch_size + 1)]
    # x_axis = np.arange(len(names))

    # plt.bar(x_axis -0.3,per_list,width=0.2,label = 'percision')
    # plt.bar(x_axis -0.1,accuracy_list,width=0.2,label = 'accuracy')
    # plt.bar(x_axis +0.1,rec_list,width=0.2,label = 'recall')
    # plt.bar(x_axis +0.3,f1_list,width=0.2,label = 'f1score')

    # plt.xticks(x_axis, names)
    # plt.legend()

    # plt.title('different statistical scores in testing')
    # plt.xlabel('testing_epoch')
    # plt.ylabel('score')
    # plt.show()


    print('Test accuracy:', np.sum(acc_list)/N,'test loss', np.sum(loss_list)/N)
    

    # list1 = pd.read_csv('./data/topk_v1_train.txt',sep=' ',header = None) #here are the code to find the common genes with and without the line 246 in pathformer.py # x = torch.mul(x, ((self.W.unsqueeze(0)).unsqueeze(2)).to(device))
    # list1 = list1[0].values.tolist()
    # list2 = pd.read_csv('./data/topk_v2_train.txt',sep=' ',header = None)
    # list2 = list2[0].values.tolist()
    # print(list(set(list1).intersection(list2)))