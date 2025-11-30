#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# # # # # # # # # # # # 
"""
╭───────────────────────────────────────╮ 
│ scRNA.py   2025/4/28/-12:04 
╰───────────────────────────────────────╯ 
│ Description:
    用于RNA数据  预处理  
""" # [By: HuYw]

# region |- Import -|
import hdf5plugin
import scanpy as sc
import pandas as pd
import os
# endregion

import argparse

exam=['-i', '/data/run01/sczd231/disk/Projects/XD/scModelMetrics/inputs/dataset/blood_immune/test.h5ad', 
'-o', '/data/run01/sczd231/disk/Projects/XD/scModelMetrics/outputs/demo/pre/Gex/test', 
'-sp', 'hg',
'-BK', 'dataset_id',
'-MK', 'cell_type', 
'-gI', "feature_id",
'-gS', "feature_name",
]

parser = argparse.ArgumentParser(description="处理输入目录和输出文件路径")
parser.add_argument("-i", "--input", type=str, help="输入Merge Raw h5ad")
parser.add_argument("-o", "--outprefix", type=str, required=True, help="输出前缀")
parser.add_argument("-BK", "--batchKey", type=str, default=None, help="obs.Batch key")
parser.add_argument("-gI", "--geneID", type=str, default="gene_ids", required=False, help="gene id var_key, 无是index")
parser.add_argument("-gS", "--geneSymbol", type=str, default=None, required=False, help="gene symbol var_key, 无是index")
parser.add_argument("-MK", "--moreKey", type=str, nargs='+', default=[], required=False, help="想绘制的更多 more obs key")
parser.add_argument("-n", "--nhvg", type=int, default=3000, help="hvg数目, 3000")
parser.add_argument("-mg", "--mingene", type=int, default=300, help="cell最小基因测量数")
parser.add_argument("-mc", "--mincell", type=int, default=3, help="gene最小细胞表达数")
parser.add_argument("-mt", "--mtCutoff", type=int, default=10, help="mt cutoff 百分比, 默认是10")
parser.add_argument("-sp", "--species", type=str, default="hg", help="物种 (hg/mm) (人类/小鼠)")
args = parser.parse_args()

os.makedirs(os.path.dirname(args.outprefix), exist_ok=True)

INFO=pd.DataFrame(index=['n.Cell', 'Median.Gene', 'Median.UMI', 'Median.MT', 'n.Cluster',
                         'n.Phase-G1', 'n.Phase-G2M', 'n.Phase-S'], columns=['Value'])
INFO['Value'] = -1

adata = sc.read(args.input)
n_obs = adata.n_obs
# 赋值gene
adata.var['raw.index']=adata.var_names.copy()
adata.var['gene_symbol'] = adata.var[args.geneSymbol].copy() if args.geneSymbol else adata.var_names.copy()
adata.var['gene_ids'] = adata.var[args.geneID].copy() if args.geneID else adata.var_names.copy()
adata.var_names = adata.var['gene_symbol'].copy()

mtprefix = 'MT-' if args.species.lower().startswith('hg') else 'mt-'
adata.var["mt"] = adata.var_names.str.startswith(mtprefix)  # 线粒体基因
sc.pp.calculate_qc_metrics(
    adata, qc_vars=["mt"], inplace=True, log1p=True
)
adata = adata[adata.obs['pct_counts_mt'] < args.mtCutoff, :]
print(f"Filter MT rate from {n_obs} -> {adata.n_obs}")

sc.pp.filter_cells(adata, min_genes=args.mingene)
sc.pp.filter_genes(adata, min_cells=args.mincell)
print(adata)
INFO.loc['n.Cell', 'Value'] = adata.n_obs
INFO.loc['Median.Gene', 'Value'] = round(adata.obs['n_genes'].median())
INFO.loc['Median.UMI', 'Value'] = round(adata.obs['total_counts'].median())
INFO.loc['Mean.MT', 'Value'] = round(adata.obs['pct_counts_mt'].mean())

# 必须是原始矩阵才可以
assert adata.X.max().is_integer(), f"[{args.input}] H5ad file not raw counts, job quit..."



# 保留raw counts
adata.layers["counts"] = adata.X.copy()
# Normalizing to median total counts
sc.pp.normalize_total(adata)
# Logarithmize the data
sc.pp.log1p(adata)

# 筛选HVG
sc.pp.highly_variable_genes(
	adata,
	n_top_genes=args.nhvg,
	layer='counts',
	flavor="seurat_v3",
	batch_key=args.batchKey,
)

# 细胞周期 基因集来自Seurat
s_genes = [
    "MCM5", "PCNA", "TYMS", "FEN1", "MCM2", "MCM4", "RRM1",
    "UNG", "GINS2", "MCM6", "CDCA7", "DTL", "PRIM1", "UHRF1",
    "HELLS", "RFC2", "RPA2", "NASP", "RAD51AP1", "GMNN", "WDR76",
    "SLBP", "CCNE2", "UBR7", "POLD3", "MSH2", "ATAD2", "RAD51",
    "RRM2", "CDC45", "CDC6", "EXO1", "TIPIN", "DSCC1", "BLM",
    "CASP8AP2", "USP1", "CLSPN", "POLA1", "CHAF1B", "BRIP1", "E2F8"
]
g2m_genes = [
    "HMGB2", "CDK1", "NUSAP1", "UBE2C", "BIRC5", "TPX2", "TOP2A", "NDC80",
    "CKS2", "NUF2", "CKS1B", "MKI67", "TMPO", "CENPF", "TACC3", "SMC4",
    "CCNB2", "CKAP2L", "CKAP2", "AURKB", "BUB1", "KIF11", "ANP32E", "TUBB4B",
    "GTSE1", "KIF20B", "HJURP", "CDCA3", "CDC20", "TTK", "CDC25C", "KIF2C",
    "RANGAP1", "NCAPD2", "DLGAP5", "CDCA2", "CDCA8", "ECT2", "KIF23", "HMMR",
    "AURKA", "PSRC1", "ANLN", "LBR", "CKAP5", "CENPE", "CTCF", "NEK2",
    "G2E3", "GAS2L3", "CBX5", "CENPA"
]
# 细胞周期小鼠化
if args.species.lower().startswith("mm"):
    s_genes = [g[0].upper() + g[1:].lower() for g in s_genes]
    g2m_genes = [g[0].upper() + g[1:].lower() for g in g2m_genes]


sc.tl.score_genes_cell_cycle(adata, s_genes=s_genes, g2m_genes=g2m_genes)
phase_df=adata.obs['phase'].value_counts().astype(int)
phase_df.index = 'n.Phase-' + phase_df.index 
INFO.loc[phase_df.index, 'Value']=phase_df.values
INFO['Value']=INFO['Value'].astype(int)

sc.tl.pca(adata)
sc.pp.neighbors(adata)
sc.tl.umap(adata)

# Using the igraph implementation and a fixed number of iterations can be significantly faster, especially for larger datasets
sc.tl.leiden(adata)
INFO.loc['n.Cluster', 'Value'] = len(adata.obs['leiden'].unique())
INFO.to_csv(f"{args.outprefix}.Metrics.tsv", sep='\t')
adata.obs.to_csv(f"{args.outprefix}.obs.tsv", sep='\t')

# QC plots umap
obs_qc = ["n_genes", "total_counts", "pct_counts_mt"]
umap = sc.pl.umap(
    adata,
    color=obs_qc,
    # Setting a smaller point size to get prevent overlap
    size=5,
    return_fig=True,
    wspace=0.5,
)
umap.savefig(f"{args.outprefix}.UMAP.QC.png", bbox_inches='tight', dpi=300)

# cate plost umap
obs_cate=["leiden", "phase"] + args.moreKey
if args.batchKey:
    obs_cate += [args.batchKey]


umap = sc.pl.umap(
    adata,
    color=obs_cate,
    # Setting a smaller point size to get prevent overlap
    size=5,
    return_fig=True,
    wspace=0.5,
)
umap.savefig(f"{args.outprefix}.UMAP.Cate.png", bbox_inches='tight', dpi=300)


# deg
import pandas as pd
obs_key='leiden'
# 计算差异基因
sc.tl.rank_genes_groups(adata, groupby=obs_key, method="wilcoxon")
# standard_scale=None 才显示真实的norm值, 否则是rank百分数
fig_dotplot = sc.pl.rank_genes_groups_dotplot(
    adata, groupby=obs_key, standard_scale=None, n_genes=5, return_fig=True
)
# categories_order 指定dotplot顺序
# dendrogram=False 顺序才生效
fig_dotplot.savefig(f"{args.outdir}/DEG.{obs_key}.png", dpi=300, bbox_inches='tight')
# 获取所有组高变基因数据框
deg_groups = adata.uns["rank_genes_groups"]["names"].dtype.names
deg_df = []
for g in deg_groups:
    tmpdf = sc.get.rank_genes_groups_df(adata, group=g)
    tmpdf["group"] = g
    tmpdf=tmpdf[['names', 'group', 'scores', 'logfoldchanges', 'pvals_adj']]
    deg_df.append(tmpdf)


# 导出deg
deg_df = pd.concat(deg_df, ignore_index=True)
deg_df.to_csv(f"{args.outprefix}.DEG.{obs_key}.tsv", sep='\t')
# 输出head5
print("Head 5 DEG for each cluster")
for g in deg_df['group'].unique():
    print(f'{g}: {deg_df[deg_df["group"]==g].sort_values("scores").head(5)["names"].tolist()}')




# 保存
adata.write(f"{args.outprefix}.h5ad")

