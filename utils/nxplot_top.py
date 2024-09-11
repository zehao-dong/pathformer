import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import numpy as np
import torch
import itertools
from matplotlib.patches import Patch
import scipy.sparse as sp
# Load gene names from txt file

#['APP', 'PSEN1', 'CTNNA2', 'PICALM'] #mayo
#['PTK2B', 'CNTNAP2', 'MAPT', 'CTNNA2', 'APOE', 'PLD3'] # rosmap
#['PTK2B', 'MAPT', 'PLD3', 'CTNNA2', 'CNTNAP2', 'APOE']
#['PICALM', 'VPS35', 'MAPT', 'MEF2C', 'CTNNA2', 'APOE', 'PSEN1'] #rosmap same genes

# with open('D:/aaai23_supp/aaai23_supp/code/AD/data_mayo.pkl', 'rb') as f:
#     loaded_data = pickle.load(f)

# with open('D:/aaai23_supp/aaai23_supp/code/AD/data_rosmap.pkl', 'rb') as f:
#     loaded_data = pickle.load(f)

with open('D:/aaai23_supp/aaai23_supp/code/AD/data_same.pkl', 'rb') as f:
    loaded_data = pickle.load(f)

loaded_data = torch.stack(loaded_data)
loaded_data = loaded_data.cpu()
summed_data = loaded_data.sum(dim=(0, 2)) # Sum along dimensions 0 and 2
final_data = summed_data.sum(dim=0).detach().numpy() # Sum along dimension 0 and convert to numpy array


# loaded_gene_names = np.load('D:/aaai23_supp/aaai23_supp/code/AD/gene_names_mayo.npy')
# loaded_gene_names = np.load('D:/aaai23_supp/aaai23_supp/code/AD/gene_names2.npy')
loaded_gene_names = np.load('D:/aaai23_supp/aaai23_supp/code/AD/gene_names_same.npy')

genes = ["APP", "PSEN2", "BIN", "CR1", "CLU", "MS4A", "PICALM", "GRN", "TSPAN14", "FNBP1L", "SEL1L", "LINC00298", "PRKCH", "C15ORF41", "C2CD3", "KIF2A", "APC", "LHX9", "NALCN", "CTNNA2", "SYTL3", "CLSTN2", "IQCK", "ACE", "ADAM10", "ADAMTS1", "WWOX", "HLA-DRB1", "TREM2", "CD2AP", "NYAP1", "EPHA1", "PTK2B", "ECHDC3", "SPI1", "MS4A2", "SORL1", "FERMT2", "SLC24A4", "ABCA7", "APOE", "MAPT", "ANKRD31", "CD33", "HBEGF", "NDUFAF6", "SCIMP", "ABI3", "AC074212.3", "ADAMTS4", "ALPK2", "APH1B", "CNTNAP2", "INPPD5", "KAT8", "MS4A6A", "ZCWPW1", "HESX1", "INPP5D", "MEF2C", "NME8", "CELF1", "FERMT2", "FRMD4A", "FBXL7", "ABI3", "PLCG2", "GGA3", "UNC5C", "PSEN1", "VPS35", "MARK4", "AKAP9", "PLD3", "NICASTRIN", "ABCA1"]
list1 = pd.read_csv('D:/aaai23_supp/aaai23_supp/code/AD/data/result/pathformer/mayo.txt',sep=' ',header = None)
list1 = list1[0].values.tolist()
# gene_names = list(set(list1).intersection(genes))
# print(gene_names)
# printtt



list2 = pd.read_csv('D:/aaai23_supp/aaai23_supp/code/AD/data/result/pathformer/rosmap.txt',sep=' ',header = None)
list2 = list2[0].values.tolist()

gene_names = list(set(list1).intersection(list2))
# print(len(gene_names))
# printtt

# Load gene pairs from txt file
with open("D:/aaai23_supp/aaai23_supp/code/AD/data/BIOGRIDALL.txt", "r") as file:
    gene_pairs = [tuple(line.strip().split("	")) for line in file]

filtered_gene_pairs = [pair for pair in gene_pairs if pair[0] in gene_names and pair[1] in gene_names and pair[0] != pair[1]]


unique_genes = list(set([gene for pair in filtered_gene_pairs for gene in pair]))
# Write gene-barcode mapping to file
# barcode_data = pd.DataFrame({'barcode': gene_names})
# barcode_data.to_csv('barcodes_same.tsv', sep='\t', index=False)
# printtt





#为了GO term
# Convert gene pairs to a DataFrame with a weight column
# edgelist = pd.DataFrame(filtered_gene_pairs, columns=['gene1', 'gene2'])
# edgelist['weight'] = 1

# # Write connectivity data to file
# edgelist.to_csv('connectivity.tsv', sep='\t', index=False)


# # Load gene expression data from file
# expression_data = pd.read_csv('D:/aaai23_supp/aaai23_supp/code/AD/data/Mayo_Exp_1a.csv', sep=' ')

# # Get gene names from first column of expression data
# all_gene_names = np.array(list(expression_data.index))

# # Find indices of genes in gene_pairs in expression data
# # Find indices of genes in gene_pairs in expression data
# selected_gene_indices = set()
# for gene_pair in filtered_gene_pairs:
#     for gene in gene_pair:
#         gene_index = np.where(all_gene_names == gene)[0]
#         if len(gene_index) > 0:
#             selected_gene_indices.add(gene_index[0])

# # Extract expression data for selected genes and transpose to have samples as rows
# selected_data = expression_data.iloc[list(selected_gene_indices), 1:].T

# # Convert selected data to sparse matrix format
# selected_sparse = sp.csr_matrix(selected_data)

# # Write selected data to file in mtx format
# sp.save_npz('gene_expression.mtx', selected_sparse)
# printt

























# Create an empty graph
graph = nx.Graph()

# Add nodes to the graph
graph.add_nodes_from(gene_names)

# Add edges to the graph
graph.add_edges_from((pair[0], pair[1]) for pair in filtered_gene_pairs)

# Draw the graph
# nx.draw(graph, with_labels=True)
non_singletons = [node for node in graph.nodes if len(graph[node]) > 0]
indexes = [i for i, gene in enumerate(loaded_gene_names) if gene in non_singletons]
print(indexes)
subgraph = graph.subgraph(non_singletons)
edges = list(subgraph.edges)
print(edges)
# Create a dictionary to store the mapping from gene names to indices
gene_to_index = {gene: index for index, gene in enumerate(loaded_gene_names)}

# Use the dictionary to look up the indices for each gene in the edges list
indexed_edges = [(gene_to_index[gene1], gene_to_index[gene2]) for gene1, gene2 in edges]
print(indexed_edges)
values = [final_data[i, j] if final_data[i, j] > final_data[j, i] else final_data[j, i] for i, j in indexed_edges]
combined = list(zip(values, edges))
combined = sorted(combined, key=lambda x: x[0], reverse=True)
top_50 = combined[:50]
top_50_edges = [x[1] for x in top_50]
print(top_50_edges)
unique_elements = len(set(top_50_edges))
print(unique_elements)
G = nx.Graph()

# Add nodes to the graph
G.add_nodes_from(non_singletons)

# Add edges to the graph
G.add_edges_from(top_50_edges)

non_single = [node for node in G.nodes if len(G[node]) > 0]

gene_names2 = list(set(non_single).intersection(genes))
print(gene_names2)



unique_elements = len(set(non_single))
print(unique_elements)
print(len(non_single))
print(non_single)

print("---------------------------------------")
subgraph2 = graph.subgraph(non_single)
print(subgraph2.edges())
# subedges = subgraph.edges()
# print(subedges)


# Find edges that connect target nodes
target_edges = [(u, v) for u, v in subgraph2.edges() if u in gene_names2 or v in gene_names2]
print("---------------------------------------222")
print(target_edges)

# # Find edges where both endpoints are in gene_names2
# both_target_edges = [(u, v) for u, v in target_edges if u in gene_names2 and v in gene_names2]
# print("---------------------------------------333")
# print(both_target_edges)


# Find all possible pairs of target genes
gene_pairs = list(itertools.combinations(gene_names2, 2))
print(gene_pairs)
print("++++++++++++++++++++++++++++++++++++++")
# Find shortest paths for all pairs of target genes
shortest_paths = {}
for pair in gene_pairs:
    shortest_path = nx.shortest_path(subgraph2, source=pair[0], target=pair[1])
    print(shortest_path)
    print("++++++++++++++++++++++++++++++++++++++")
    for i in range(len(shortest_path)-1):
        edge = (shortest_path[i], shortest_path[i+1])
        shortest_paths[edge] = 'green'

print(shortest_paths)
print("++++++++++++++++++++++++++++++++++++++")

target_nodes = list(set([v for edge in target_edges for v in edge if v not in gene_names2]))
print(target_nodes)
print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")

unique_nodes2 = list(set([node for edge in shortest_paths.keys() for node in edge if node not in target_nodes and node not in gene_names2]))

print(unique_nodes2)

unique_nodes3 = list(set(unique_nodes2).intersection(target_nodes))



print(list(shortest_paths.keys()))
print('/////////////////////////////////////')
for edge in subgraph2.edges():
    if edge in target_edges and edge not in list(shortest_paths.keys()):
        print(edge)

node_colors = ['red' if node in gene_names2 else 'orange' if node in target_nodes or node in unique_nodes2 else 'lightblue' for node in subgraph2.nodes()]

# Define edge colors, thicker for target and both target edges
edge_colors = ['green' if ((edge[0],edge[1]) in list(shortest_paths.keys())) or ((edge[1],edge[0]) in list(shortest_paths.keys())) else 'darkblue' if ((edge[0],edge[1]) in target_edges or (edge[1],edge[0]) in target_edges) and ((edge[0],edge[1]) not in list(shortest_paths.keys()) and (edge[1],edge[0]) not in list(shortest_paths.keys())) else 'black' for edge in subgraph2.edges()]

# Define edge widths
edge_widths = [3 if ((edge[0],edge[1]) in target_edges + list(shortest_paths.keys())) or ((edge[1],edge[0]) in target_edges + list(shortest_paths.keys())) else 0.1 for edge in subgraph2.edges()]

# node_colors = [subgraph.degree[node] for node in subgraph]
pos = nx.kamada_kawai_layout(subgraph2, scale=2000) #k=2, iterations=200


# nx.draw(subgraph, pos, with_labels=True, node_color=node_colors, cmap=plt.cm.viridis)
nx.draw(subgraph2, pos, node_size=500, with_labels=True, edge_color=edge_colors, node_color=node_colors, font_size=16, font_weight='bold', width=edge_widths)

green_patch = plt.Line2D([], [], color='green', label='Shortest path')
darkblue_patch = plt.Line2D([], [], color='darkblue', label='1-hop edges')
lightblue_patch = plt.scatter([],[],color='lightblue', label='Possible unimportant genes')
red_patch = plt.scatter([],[],color='red', label='Known AD genes and risk gene variants')
orange_patch = plt.scatter([],[],color='orange', label='Possible important genes on shortest path and 1-hop away')

plt.legend(handles=[green_patch, darkblue_patch, lightblue_patch, red_patch, orange_patch], loc='lower left', fontsize=16)
plt.show()



# Draw the graph
# Find corresponding edge values in final_data for subgraph2 and store in graph_edge_values with their corresponding pairs in graph_edge_pairs
# graph_edge_pairs = []
# graph_edge_values = []
# for edge in subgraph2.edges():
#     gene1, gene2 = edge[0], edge[1]
#     gene1_idx = np.where(loaded_gene_names == gene1)[0][0]
#     gene2_idx = np.where(loaded_gene_names == gene2)[0][0]
#     edge_value = final_data[gene1_idx, gene2_idx]
#     graph_edge_values.append(edge_value)
#     rank = np.sum(final_data > edge_value) + 1
#     graph_edge_pairs.append((gene1, gene2, rank, edge_value))

# # Sort the graph edge values and pairs in descending order
# graph_edge_pairs_descending_ranked = sorted(graph_edge_pairs, key=lambda x: x[3], reverse=True)
# print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
# print(graph_edge_pairs_descending_ranked)