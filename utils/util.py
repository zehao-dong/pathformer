import numpy as np
import igraph as ig
import pandas as pd
from scipy import sparse
import torch
import csv
from tqdm import tqdm
import random
import torch.nn.functional as F

def load_biogrid_df(name, pwd, colnames = ['Source','Target']):
    di_edges = []
    with open('%s/%s.txt' % (pwd, name), 'r') as f:
        #print(f)
        for i,row in enumerate(f):
            if i >= 1:
                row_split = row.split()
                di_edges.append([row_split[0],row_split[1]])
                #di_edges[1].append(row[1])
    df_edges = pd.DataFrame(di_edges,columns=colnames)
    return df_edges


def ad_preprocess(data, K):
    var = np.var(data.values,1)#axis=1，沿行的方差
    columns = list(data.columns)
    y = np.array([1 if x.startswith('control') else 0 for x in columns])#y贴标签1or0
    data['var'] = var
    sort_data = data.sort_values(by='var',ascending=False)
    filt_data = sort_data[:K]#去掉var变化值小的数,只取前3000个
    print("3000 selected genes based on variance")
    print(filt_data)

    # Create a new data frame with column names corresponding to y
    new_data = filt_data.copy()
    # Drop the 'var' column from the new data frame
    new_data = new_data.drop(columns=['var'])

    return filt_data, y, new_data

def ad_preprocess2(data, K):
    var = np.var(data.values,1)#axis=1，沿行的方差
    columns = list(data.columns)
    y = np.array([1 if x.startswith('control') else 0 for x in columns])#y贴标签1or0
    data['var'] = var
    sort_data = data.sort_values(by='var',ascending=False)
    filt_data = sort_data[:K]
    # Create a new data frame with column names corresponding to y
    new_data = filt_data.copy()
    # Drop the 'var' column from the new data frame
    new_data = new_data.drop(columns=['var'])
    return new_data

def remove_samegene(data, dataset):
    mask = np.isin(dataset.index, data)
    dataset.drop(dataset[mask].index, inplace=True)
    return dataset
def get_samegene(data, dataset):
    mask = np.isin(dataset.index, data)
    return dataset.loc[mask].copy()


def log2fc(dataset, y):
    
    control_cols = [col for col in dataset.columns if col.startswith('control')]
    disease_cols = [col for col in dataset.columns if col.startswith('AD')]

    control_values = dataset[control_cols].values
    disease_values = dataset[disease_cols].values

    # Calculate the mean expression for each condition
    mean_control = np.mean(control_values, axis=1)  # Mean expression of control samples for all genes
    mean_disease = np.mean(disease_values, axis=1)  # Mean expression of disease samples for all genes

    # Calculate the fold change for each gene
    fold_change = np.abs(np.log2(mean_disease / mean_control))

    return fold_change

def log2fc_back(batch_x, batch_y, P, indices, device, scale):

    b_size = batch_x.size(0)
    weight =torch.stack([P]*b_size, dim=0).to(device)
    val = torch.mul(batch_x, weight).index_select(1, indices.to(device))
    all_val = torch.sum(val, dim=1)
    control_values = [all_val[i] for i in range(b_size) if batch_y[i].item() < 0.5]
    disease_values = [all_val[i] for i in range(b_size) if batch_y[i].item() > 0.5]
    if len(control_values) > 0 and len(disease_values) > 0:
        mean_control = torch.mean(torch.stack(control_values)) + 0.00001 # Mean expression of control samples for all genes
        mean_disease = torch.mean(torch.stack(disease_values)) + 0.00001 # Mean expression of disease samples for all genes

        # Calculate the fold change for each gene
        fold_change = torch.abs(torch.log2(mean_disease / mean_control))
    else:
        fold_change = torch.Tensor([0]).to(device)

    return fold_change * scale


def regular_term(cur_loss, cur_p, pre_loss, pre_p, device, scale=1, reduc='mean'):
    
    kl_loss = - F.kl_div(cur_p,pre_p, reduction=reduc)
    loss_diff = cur_loss - pre_loss
    reg = torch.Tensor([0]).to(device)
    if loss_diff.item() < 0:
        #reg += kl_loss/(-loss_diff + 1)
        reg += kl_loss/(-loss_diff * 5 + 1)
    return scale * reg


def adj_generate(filt_data,biogrid_intact,K, reverse = False):
    adj = np.zeros((K,K))#3000*3000的一个adjacency matrix初始化都为0
    gene_list = list(filt_data.index)#取data的第一column的element，不取第一column的header
    gene_map = {gene: i for i,gene in zip(range(K),gene_list)} #算是一个dictionary

    N = biogrid_intact.shape[0]#472643 rows * 2 columns, [0]:472643 rows
    pbar = tqdm(range(N))#可以在 Python 长循环中添加一个进度提示信息

    for i in pbar:
        gene1, gene2 = biogrid_intact.iloc[i]['Source'], biogrid_intact.iloc[i]['Target']
        if gene1 in gene_list:
            if gene2 in gene_list:
                idx1, idx2 = gene_map[gene1], gene_map[gene2]#在dictionary里找对应的value，也就是index

                adj[idx1, idx2] = 1#把对应的adjacency matrix坐标点从0改成1
                if reverse:#暂时不理解
                    adj[idx2, idx1] = 1

    return adj

def adj_generate2(filt_data,biogrid_intact,K, reverse = False):
    adj = np.zeros((K,K))#3000*3000的一个adjacency matrix初始化都为0
    gene_list = list(filt_data.index)#取data的第一column的element，不取第一column的header
    gene_map = {gene: i for i,gene in zip(range(K),gene_list)} #算是一个dictionary
    N = biogrid_intact.shape[0]#472643 rows * 2 columns, [0]:472643 rows
    textfile = open("D:/aaai23_supp/aaai23_supp/code/AD/data/Rosmap_A2.txt", "w")

    for i in range(N):
        gene1, gene2 = biogrid_intact.iloc[i]['Source'], biogrid_intact.iloc[i]['Target']
        if gene1 in gene_list:
            if gene2 in gene_list:
                idx1, idx2 = gene_map[gene1], gene_map[gene2]#在dictionary里找对应的value，也就是index
                if idx1 != idx2:
                    textfile.write(str(idx1)+ "," + " " + str(idx2) + "\n")

    textfile.close()
    return adj



def find_indices(list_to_check, item_to_find):
    indices = []
    for idx, value in enumerate(list_to_check):
        if value == item_to_find:
            indices.append(idx)
    return indices


def adj_gnn(filt_data,biogrid_intact,K):
    adj = []
    gene_list = list(filt_data.index)#取data的第一column的element，不取第一column的header
    gene_map = {gene: i for i,gene in zip(range(K),gene_list)} #算是一个dictionary
    source = biogrid_intact['Source'].tolist()
    target = biogrid_intact['Target'].tolist()

    pbar = tqdm(range(len(gene_list)))
    for i in pbar:
        if gene_list[i] in source:
            indices = find_indices(source, gene_list[i])
            for index in indices:
                if target[index] in gene_map.keys():
                    adj.append([gene_map[gene_list[i]], gene_map[target[index]]])
        if gene_list[i] in target:
            indices = find_indices(target, gene_list[i])
            for index in indices:
                if source[index] in gene_map.keys():
                    adj.append([gene_map[gene_list[i]], gene_map[source[index]]])            
    return adj

def dataset_prepare2(X, Y, fold_id, fold=5):    
    #X = torch.FloatTensor(X.values).permute(1,0)[:-1]#将tensor的维度换位 ex  3*2 to 2*3
    #Y = torch.LongTensor(Y)
    #N = X.size(0)-1 #158
    N = X.size(0)
    fold_num = int(np.ceil(N / fold))

    train_idx = torch.LongTensor([1]*N) 
    test_idx = torch.LongTensor([0]*N) 

    min_idx = fold_id * fold_num
    max_idx = min((fold_id+1) * fold_num, N)-1

    for i in range(N):
        if i <= max_idx and i >= min_idx:
            train_idx[i] = 0
            test_idx[i] = 1
    
    train_idx = train_idx.to(torch.bool)
    test_idx = test_idx.to(torch.bool)
  

    X_train = X[train_idx] #N*p = 158*0.8 = 126.4,取前多少行rows
    Y_train = Y[train_idx]
    X_test = X[test_idx]
    Y_test = Y[test_idx]
    
    return X_train, Y_train, X_test, Y_test




def dataset_prepare(X, Y, p=0.8):    
    X = torch.FloatTensor(X.values).permute(1,0)[:-1]#将tensor的维度换位 ex  3*2 to 2*3
    Y = torch.LongTensor(Y)
    N = X.size(0)-1 #158
    idx = list(range(N))
    
    random.shuffle(idx)


    
    X = X[torch.LongTensor(idx)]#根据idx乱序排列X里的rows
    Y = Y[torch.LongTensor(idx)]#同理
  

    X_train = X[:int(N * p)] #N*p = 158*0.8 = 126.4,取前多少行rows
    Y_train = Y[:int(N * p)]
    X_test = X[int(N * p):]
    Y_test = Y[int(N * p):]
    
    return X_train, Y_train, X_test, Y_test