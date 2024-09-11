import pandas as pd
import numpy as np
import scipy.sparse as sp
from goatools.obo_parser import GODag
from goatools.associations import read_ncbi_gene2go
from goatools.go_enrichment import GOEnrichmentStudy
from goatools.base import download_ncbi_associations
from goatools.anno.genetogo_reader import Gene2GoReader
from goatools.base import download_go_basic_obo
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
import seaborn as sns
import textwrap

genes = pd.read_csv('D:/aaai23_supp/aaai23_supp/code/AD/Go/gene_result.txt', sep='\t')
# # Load gene expression data
# expression_data = sp.load_npz('D:/aaai23_supp/aaai23_supp/code/AD/Go/gene_expression.mtx.npz')
gene_names = pd.read_csv('D:/aaai23_supp/aaai23_supp/code/AD/Go/barcodes_same.tsv', sep='\t')['barcode'].values

# create a dictionary mapping gene symbols to GeneIDs
symbol2id = dict(zip(genes['Symbol'], genes['GeneID']))

# get the GeneIDs for the genes in gene_names
gene_ids = [symbol2id.get(name) for name in gene_names]

# Load gene-GO term associations from NCBI
gene2go_file = download_ncbi_associations()
taxid = 9606  # Human tax ID
go2geneids_human = Gene2GoReader(gene2go_file, taxids=[taxid])

# Load the Gene Ontology (GO) hierarchy from the OBO file
obo_file = download_go_basic_obo()
go_dag = GODag("go-basic.obo")

# Perform the GO analysis
pop = go2geneids_human.get_ns2assc()

goeaobj = GOEnrichmentStudy(
    genes['GeneID'],  # Study genes
    pop['BP'],  # Population genes
    go_dag,  # GO hierarchy
    propagate_counts=False,  # Set to True or False
    alpha=0.9,
    methods=['fdr_bh']
)
#run one time to initialize
GO_items = []

temp = goeaobj.assoc
for item in temp:
    GO_items += temp[item]



# Run the GO enrichment analysis using the GO IDs of the population genes
goea_results_all = goeaobj.run_study(gene_ids)
goea_results_sig = [r for r in goea_results_all if r.p_fdr_bh < 0.9]# Get significantly enriched results

GO = pd.DataFrame(list(map(lambda x: [x.GO, x.goterm.name, x.goterm.namespace, x.p_uncorrected, x.p_fdr_bh,\
                   x.ratio_in_study[0], x.ratio_in_study[1],GO_items.count(x.GO),\
                   ], goea_results_sig)), columns = ['GO', 'term', 'class', 'p', 'p_corr', 'n_genes',\
                                                    'n_study', 'n_go'])
df = GO[GO.n_genes > 1]
df['per'] = df.n_genes/df.n_go
df = df[0:10]
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
import seaborn as sns
import textwrap

fig, ax = plt.subplots(figsize = (8, 6))
sns.set(font_scale=1.7) # adjust the font scale
cmap = mpl.cm.bwr_r
norm = mpl.colors.Normalize(vmin = df.p_corr.min(), vmax = df.p_corr.max())

mapper = cm.ScalarMappable(norm = norm, cmap = cm.bwr_r)

cbl = mpl.colorbar.ColorbarBase(ax, cmap = cmap, norm = norm, orientation = 'vertical')
plt.figure(figsize = (2,4))



ax = sns.barplot(data = df, x = 'per', y = 'term', palette = mapper.to_rgba(df.p_corr.values))

# Set axis labels
ax.set_xlabel('Percentage of genes')
ax.set_ylabel('GO terms')

# Wrap the y-axis labels to fit in the plot
ax.set_yticklabels([textwrap.fill(e, 22) for e in df['term']])

plt.show()
