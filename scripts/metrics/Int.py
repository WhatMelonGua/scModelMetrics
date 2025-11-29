#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# # # # # # # # # # # # 
"""
╭───────────────────────────────────────╮ 
│ Int.py   2025/11/29/-01:00 
╰───────────────────────────────────────╯ 
│ Description:
    用于 多模态/批次 整合评估    
""" # [By: HuYw]

# region |- Import -|
import pandas as pd
import anndata as ad
from sklearn.metrics import accuracy_score, f1_score # 评估方法
import scib
import argparse
# endregion

parser = argparse.ArgumentParser(description="Run scib metrics with customizable options.")

# Add arguments for metrics to run
parser.add_argument('-bio','--bio_metrics', nargs='+', type=str.lower, help="List of metrics to run (e.g., ari, cell_cycle).")
parser.add_argument('-batch', '--batch_metrics', nargs='+', type=str.lower, help="List of metrics to skip (e.g., ari, cell_cycle).")
# Add arguments for basic input data
parser.add_argument('-a', '--adata_pre', type=str, required=True, help="Path to the pre-integration AnnData file.")
parser.add_argument('-b','--adata_post', type=str, required=True, help="Path to the post-integration AnnData file.")
parser.add_argument('-bk', '--batch_key', type=str, required=True, help="Batch key in adata.obs.")
parser.add_argument('-lk', '--label_key', type=str, required=True, help="Label key in adata.obs.")

args = parser.parse_args()

# Load data
adata_pre = ad.read_h5ad(args.adata_pre)
adata_post = ad.read_h5ad(args.adata_post)

# Define all available biological conservation metrics
biological_conservation_metrics = {
    'ari': scib.metrics.ari,
    'cell_cycle': scib.metrics.cell_cycle,
    'clisi_graph': scib.metrics.clisi_graph,
    'hvg_overlap': scib.metrics.hvg_overlap,
    'isolated_labels_asw': scib.metrics.isolated_labels_asw,
    'isolated_labels_f1': scib.metrics.isolated_labels_f1,
    'nmi': scib.metrics.nmi,
    'silhouette': scib.metrics.silhouette,
    'trajectory_conservation': scib.metrics.trajectory_conservation
}

# Define all available batch correction metrics
batch_correction_metrics = {
    'graph_connectivity': scib.metrics.graph_connectivity,
    'ilisi_graph': scib.metrics.ilisi_graph,
    'kBET': scib.metrics.kBET,
    'pcr_comparison': scib.metrics.pcr_comparison,
    'silhouette_batch': scib.metrics.silhouette_batch
}

# Combine all metrics
all_metrics = {**biological_conservation_metrics, **batch_correction_metrics}

# Determine which metrics to run
if args.run_metrics:
    metrics_to_run = {metric: all_metrics[metric] for metric in args.run_metrics if metric in all_metrics}
else:
    metrics_to_run = all_metrics

if args.skip_metrics:
    for metric in args.skip_metrics:
        if metric in metrics_to_run:
            del metrics_to_run[metric]

# Run the selected metrics
results = {}
for metric_name, metric_func in metrics_to_run.items():
    try:
        if metric_name in ['cell_cycle', 'trajectory_conservation', 'hvg_overlap']:
            result = metric_func(adata_pre, adata_post, batch_key=args.batch_key)
        elif metric_name in ['clisi_graph', 'ilisi_graph']:
            result = metric_func(adata_post, label_key=args.label_key)
        elif metric_name in ['kBET']:
            result = metric_func(adata_post, batch_key=args.batch_key, label_key=args.label_key)
        elif metric_name in ['pcr_comparison']:
            result = metric_func(adata_pre, adata_post, covariate=args.batch_key)
        elif metric_name in ['silhouette_batch']:
            result = metric_func(adata_post, batch_key=args.batch_key)
        elif metric_name in ['isolated_labels_asw', 'isolated_labels_f1']:
            result = metric_func(adata_post, label_key=args.label_key)
        elif metric_name in ['silhouette']:
            result = metric_func(adata_post, label_key=args.label_key)
        else:
            result = metric_func(adata_post, cluster_key=args.label_key, label_key=args.label_key)
        
        results[metric_name] = result
    except Exception as e:
        print(f"Error running {metric_name}: {e}")

# Print results
for metric_name, result in results.items():
    print(f"{metric_name}: {result}")
