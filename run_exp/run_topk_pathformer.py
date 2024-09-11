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
from pathformer_topk import topK_Pathormer_reverse

# PARSE ARGUMENTS FROM COMMAND LINE
def arg_parse():
    parser = argparse.ArgumentParser(description='Argparser for CapsGNN')
    # ADD FOLLOWING ARGUMENTS
    parser.add_argument('--cuda', dest = 'cuda',
                help = 'CUDA.')
    parser.add_argument('--mode', dest = 'mode',
                help = 'cpu or gpu')
    parser.add_argument('--dropout', type=float, default=0.35)
    parser.add_argument('--n_copy', type=int, default=6, help='6: catogories of copy number')
    parser.add_argument('--copy_dim', type=int, default=20, help='20: dimension of catogory')
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--d_model', type=int, default=8)
    parser.add_argument('--n_head', type=int, default=2)
    parser.add_argument('--d_ff', type=int, default=16)
    parser.add_argument('--n_nodes', type=int, default=3000, help='3000: preprocessing 3000 genes according to variance')
    parser.add_argument('--fold', type=int, default=5, help='3000: preprocessing 3000 genes according to variance')
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--out_hidden', type=int, default=128)
    parser.add_argument('--out_dim', type=int, default=2)
    parser.add_argument('--seed', type=int, default=0)#np.random.randint(1,high=10000,size=1)[0]
    parser.add_argument('--reg', action='store_true', default=False, 
                    help='use resistance distance as additional continuous node labels')
    parser.add_argument('--reg_scale', type=float, default=10)
    parser.add_argument('--K', type=int, default=1000,help='1000: selected top K genes')
    parser.add_argument('--lr', dest = 'learning_rate', type = float, default=0.001,
                help = '0.001: Learning rate.')
    parser.add_argument('--batch-size', dest = 'batch_size', type = int, default=8,
                help = 'Batch size.')
    parser.add_argument('--epochs', dest = 'num_epochs', type = int, default=150,
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
    #os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    if args.mode == 'gpu':
        device = torch.device('cuda:{}'.format(args.cuda))
    else:
        device = torch.device("cpu")
    # device = torch.device("cuda:{}".format(0))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    biogrid_intact = load_biogrid_df('BIOGRIDALL','./data')
    tgt_ad_genes = ['APOE','SORL','CLU', 'APP', 'PSEN1'， 'PSEN2', 'PICALM', 'CTNNA2', 'PTK2B', 'MAPT', 'PLD3', 'CTNNA2', 'CNTNAP2','CR']
    core_gene = ['APOE','PSEN1', 'PICALM', 'CTNNA2', 'PTK2B', 'CNTNAP2']


    
    if args.dataset == "mayo":
        mayo = pd.read_csv('./data/Mayo_Exp_1a.csv',sep=' ')
        # with open('D:/aaai23_supp/aaai23_supp/code/AD/data/result/test1/test4/combined.txt', 'r') as f:
        #     genes_to_remove = f.read().splitlines()

        # remove_samegene(genes_to_remove, mayo)
        # print(mayo)
        # print("-----------------------------------")
        raw_data, raw_y, _ = ad_preprocess(mayo,args.n_nodes)
        new_candi_genes = []
        for gene in tgt_ad_genes:
            if gene not in raw_data.index:
                print(gene)
                if gene in mayo.index:
                    new_candi_genes.append(gene)
        pad_df = mayo.loc[new_candi_genes]
        raw_data = pd.concat([raw_data, pad_df],axis=0)
        Ngenes = raw_data.shape[0]
        gene_list = list(raw_data.index) #show top 3000 gene names based on variance
    elif args.dataset == "rosmap":
        mayo = pd.read_csv('./data/Mayo_Exp_1a.csv',sep=' ')
        rosmap = pd.read_csv('./data/RosMap_Exp_1a.csv',sep=' ')
        data_mayo, _, _ = ad_preprocess(mayo, args.n_nodes)
        var = np.var(rosmap.values,1)#axis=1，沿行的方差
        columns = list(rosmap.columns)
        raw_y = np.array([1 if x.startswith('control') else 0 for x in columns])#y贴标签1or0
        rosmap['var'] = var
        raw_data = rosmap.loc[data_mayo.index]
        print(raw_data)
        #raw_data, raw_y, _ = ad_preprocess(rosmap,args.n_nodes)
        new_candi_genes = []
        for gene in tgt_ad_genes:
            if gene not in raw_data.index:
                print(gene)
                if gene in rosmap.index:
                    new_candi_genes.append(gene)
        pad_df = rosmap.loc[new_candi_genes]
        raw_data = pd.concat([raw_data, pad_df],axis=0)
        Ngenes = raw_data.shape[0]
        gene_list = list(raw_data.index) #show top 3000 gene names based on variance

    
    # mayo = pd.read_csv('./data/Mayo_Exp_1a.csv',sep=' ')
    # mayo1 = pd.read_csv('./data/RosMap_Exp_1a.csv',sep=' ')

    # data_mayo, y_mayo = ad_preprocess(mayo,args.n_nodes)
    # gene_list = list(data_mayo.index)

    #----2-----
    # new_df = get_samegene(gene_list, mayo1)
    # print("got 3000 same genes")
    # data_mayo, y_mayo = ad_preprocess(new_df,args.n_nodes)
    # gene_list2 = list(data_mayo.index)


    # textfilet = open("./data/rosgene.txt", "w")
    # for name in gene_list:
    #     textfilet.write(str(name) + "\n")
    # textfilet.close()
 
    #adj_mayo = adj_generate(raw_data,biogrid_intact,args.n_nodes, reverse = False)#make adj matrix 3000*3000
    adj_mayo = adj_generate(raw_data,biogrid_intact,Ngenes, reverse = False)#make adj matrix 3000*3000


    # X_train, Y_train, X_test, Y_test = dataset_prepare(data_mayo, y_mayo)

    for fold_idx in range(args.fold-1):
        print('fold id: {}'.format(fold_idx))
        X = torch.FloatTensor(raw_data.values).permute(1,0)[:-1]#将tensor的维度换位 ex  3*2 to 2*3
        Y = torch.LongTensor(raw_y)
        N = X.size(0)-1 #158 remove the var column
        idx = list(range(N))       
        random.shuffle(idx)      
        X = X[torch.LongTensor(idx)]#根据idx乱序排列X里的rows
        Y = Y[torch.LongTensor(idx)]#同理
        X_train, Y_train, X_test, Y_test = dataset_prepare2(X, Y, fold_idx, args.fold)
        # print(X_train.shape) (126*3000)
        # print(Y_train.shape) (126)
        # printt
        #num_sample = data_mayo.shape[0]#3000
        #res_dir = './mayo_results'
        if args.reg:
            res_dir = './' + args.dataset + '/results_{}_reg'.format(fold_idx)
        else:
            res_dir = './' + args.dataset + '/results_{}'.format(fold_idx)
        if not os.path.exists(res_dir):
            os.makedirs(res_dir) 
        

        pathformer = topK_Pathormer_reverse(n_copy=args.n_copy, copy_dim=args.copy_dim, d_model=args.d_model, n_head=args.n_head, d_ff=args.d_ff, drop_out=args.dropout, n_layers=args.n_layers, hidden_dim=args.hidden_dim,
                        out_hidden=args.out_hidden, Nnodes=Ngenes, out_dim=args.out_dim, device=device,K=args.K)
        




        loss_func =nn.CrossEntropyLoss()
        #loss_func =nn.CrossEntropyLoss(reduction='sum')
        loss_name = os.path.join(res_dir, 'train_loss_{}.txt'.format(args.K))
        optimizer = optim.Adam(pathformer.parameters(),lr=args.learning_rate)
        pathformer.to(device)

        # Train the model
        EPOCH = args.num_epochs
        batch_size = args.batch_size
        #pbar = tqdm(range(EPOCH))
        pathformer.train()
        # graphnn.train()

        per_list1 = []
        accuracy_list1 = []
        rec_list1 = []
        f1_list1 = []
        reg_list = []
        

        pre_p = torch.Tensor([1] * args.n_nodes)
        pre_p = torch.exp(pre_p)/ torch.sum(torch.exp(pre_p))
        pre_p = pre_p.to(device)
        pre_loss = torch.Tensor([1000]).to(device)

        for i in range(1,EPOCH+1):
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

                pred, indexes, attn_list, cur_p = pathformer(batch_x.unsqueeze(2), A)

                nppred1 = (pred.cpu().detach().numpy()).tolist()
                nppred1 = [p.index(max(p)) for p in nppred1]
                fin_nppred.extend(nppred1)

                npbatch_y1 = (batch_y.cpu().detach().numpy()).tolist()
                fin_npbatch_y.extend(npbatch_y1)
                
                
                loss_ = loss_func(pred, batch_y) # Computes the loss between the predicted values (pred) and the actual target values (batch_y)

                #reg = regular_term(loss_.detach(), cur_p, pre_loss, pre_p, device, scale=0.1, reduc='sum')
                if args.reg:
                    reg = log2fc_back(batch_x, batch_y, cur_p, indexes, device, scale=args.reg_scale)

                    #loss = loss_ + reg
                    loss = loss_ + reg
                else:
                    reg = torch.Tensor([0])
                    loss = loss_

                # Zeroes out the gradients computed during the previous backward pass. 
                # This is necessary because PyTorch accumulates gradients by default
                # and calling zero_grad() ensures that the gradients for the current batch are the only ones used in the current optimization step
                optimizer.zero_grad()
                loss.backward() # Computes gradients of the loss with respect to the model parameters
                optimizer.step() # Updates the model parameters based on the computed gradients and the chosen optimization algorithm.
                target_true = torch.sum(torch.argmax(pred, dim=-1) == batch_y).item() # Computes the number of correct predictions in the current batch.
                acc = target_true / B # Computes the accuracy of the model on the current batch.
                acc_list.append(target_true) # Appends the number of correct predictions to the list of accuracies for the current epoch.
                total_B += B # Updates the total number of samples processed so far in the current epoch
                # Appends the value of the loss function multiplied by the batch size to the list of losses for the current epoch.
                loss_list.append(loss.item() * B)
                reg_list.append(reg.item() * B)

                # Sets the description for the progress bar displayed during training. 
                # The description includes the epoch number, current loss, current accuracy, and number of correct predictions.
                pbar.set_description('Epoch: {0} total_loss:{1:.3f} acc: {2:.4f}, class_loss {3:.4f}, , reg_loss {4:.4f}'.format(i, loss.item(), acc, loss_.item(), reg.item()))
                #pre_loss = loss_.detach()
                #pre_p = cur_p.detach()
            
            
            
            

            #if i % 50 == 0: # Checks whether the total number of processed samples is divisible by 50
            if i == EPOCH:
                print("save current model...")
                # Creates a file path for saving the current model checkpoint, including the current epoch number in the filename
                model_name = os.path.join(res_dir, 'model_{}_checkpoint{}.pth'.format(args.K,i))
                # Creates a file path for saving the current optimizer checkpoint, including the current epoch number in the filename.
                optimizer_name = os.path.join(res_dir, 'optimizer_{}_checkpoint{}.pth'.format(args.K,i))
                
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
            
            with open(loss_name, 'a') as loss_file:
                loss_file.write("{:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f}\n".format(
                    np.sum(acc_list)/total_B, 
                    np.sum(loss_list)/total_B, 
                    np.sum(reg_list)/total_B, 
                    per_score, 
                    rec_score, 
                    f_score
                    ))
            

            print('Epoch:', i ,'Train accuracy:', np.sum(acc_list)/total_B, 'train loss', np.sum(loss_list)/total_B)
            print('f1_score:', f_score, 'percision', per_score, 'recall', rec_score)
        # torch.save(pathformer,'model2.pth')
        #after training, add topk genes into a txt file
        # print(len(attn_list))
        # print(len(attn_list[0]))
        # print(len(attn_list[0][0]))
        # print(len(attn_list[0][0][0]))
        # print(len(attn_list[0][0][0][0]))
        # print(len(attn_list[0][0][0][0][0]))

        # Save the attn_list to a file
        # with open('D:/aaai23_supp/aaai23_supp/code/AD/data/result/test1/test3/data_same_18.pkl', 'wb') as f:
        #     pickle.dump(attn_list, f)

        #textfile2 = open("./data/result/test1/test4/same_16.txt", "w")#save idx of Top K genes 
        #for idx in indexes:
        #    if args.dataset == "same":
        #        textfile2.write(str(gene_list2[idx]) + "\n")# if mayo, gene_list, if rosmap, gene_list2
        #    else:
        #        textfile2.write(str(gene_list[idx]) + "\n")
        #textfile2.close()

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

        pathformer.eval()
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
            # print(adj_mayo.shape)
            # print(len([torch.FloatTensor(adj_mayo)]))
            # print(len([torch.FloatTensor(adj_mayo)][0]))
            # print(len([torch.FloatTensor(adj_mayo)][0][0]))
            # print([torch.FloatTensor(adj_mayo)][0][0][0])
            # print("------------------------------")
            # print(len([torch.FloatTensor(adj_mayo)]*B))
            # print(len(([torch.FloatTensor(adj_mayo)]*B)[0]))
            # print(len(([torch.FloatTensor(adj_mayo)]*B)[0][0]))
            # print(([torch.FloatTensor(adj_mayo)]*B)[0][0][0])
            A = torch.stack([torch.FloatTensor(adj_mayo)] * B, dim=0)
            A = A.to(device)
            with torch.no_grad():
                pred, indexes, attn_list, _ = pathformer(batch_x.unsqueeze(2), A)
                loss = loss_func(pred, batch_y)

            nppred = (pred.cpu().detach().numpy()).tolist()
            nppred = [p.index(max(p)) for p in nppred]
            

            npbatch_y = (batch_y.cpu().detach().numpy()).tolist()
            

            target_true = torch.sum(torch.argmax(pred, dim=-1) == batch_y).item()
            acc_list.append(target_true)
            # total_B += B
            loss_list.append(loss.item() * B)
            nppred1 = (pred.cpu().detach().numpy()).tolist()
            nppred1 = [p.index(max(p)) for p in nppred1]
            fin_nppred.extend(nppred1)

            npbatch_y1 = (batch_y.cpu().detach().numpy()).tolist()
            fin_npbatch_y.extend(npbatch_y1)
            # pbar.set_description('Epoch: {0} loss:{1:.3f} acc: {2:.4f}, :num: {3}'.format(i, loss.item(), acc, target_true))

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
        # textfile2 = open("./data/topk_v6_test.txt", "w")
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
        test_results_name = os.path.join(res_dir, 'test_results_{}.txt'.format(args.K))
        per_score = precision_score(fin_npbatch_y, fin_nppred, average='weighted')
        acc_score = np.sum(acc_list)/N
        rec_score = recall_score(fin_npbatch_y, fin_nppred, average='weighted')
        f_score = f1_score(fin_npbatch_y, fin_nppred, average='weighted')
        #acc_score = accuracy_score(fin_npbatch_y, fin_nppred)

        #print('Test accuracy:', np.sum(acc_list)/N,'test loss', np.sum(loss_list)/N)
        print('Test accuracy:', acc_score,'test loss', np.sum(loss_list)/N)
        print('f1_score:', f_score, 'percision', per_score, 'recall', rec_score)
        indexes_string = str(indexes.cpu().detach().numpy().tolist())
        with open(test_results_name, 'a') as result_file:
            result_file.write("Test accuracy: {:.4f} test loss: {:.4f} ".format(
                    acc_score, np.sum(loss_list)/N) + 
                    " per_score: {:.4f}  rec_score: {:.4f} f_score: {:.4f}\n".format(per_score,
                   rec_score, f_score))
            result_file.write("Test accuracy" + indexes_string + "\n")
        # np.sum(acc_list)/N

        # list1 = pd.read_csv('./data/topk_v1_train.txt',sep=' ',header = None) #here are the code to find the common genes with and without the line 246 # x = torch.mul(x, ((self.W.unsqueeze(0)).unsqueeze(2)).to(device))
        # list1 = list1[0].values.tolist()
        # list2 = pd.read_csv('./data/topk_v2_train.txt',sep=' ',header = None)
        # list2 = list2[0].values.tolist()
        # print(list(set(list1).intersection(list2)))