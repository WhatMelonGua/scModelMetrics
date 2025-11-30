#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# # # # # # # # # # # # 
"""
╭───────────────────────────────────────╮ 
│ 2.doMetrics.smk   2025/11/28/-14:59 
╰───────────────────────────────────────╯ 
│ Description:
    指标评估 规则    
""" # [By: HuYw]

# region |- Import -|
from types import SimpleNamespace
import os
# endregion

rule MetricAnno:
    input:
        anno="{outdir}/callModel/{model}/{dataname}/anno.tsv",  # 假设这是注释结果文件
    output:
        metric="{outdir}/callModel/{model}/metrics/{dataname}/anno.tsv",  # 输出的评估结果文件
    log:
        "{outdir}/log/2.Anno.log"
    benchmark:
        "{outdir}/benchmark/2.Anno.benchmark"
    shell:
        """
        python scripts/metrics/Anno.py \
          -i {input.anno} \
          -o {output.metric} &>> {log}
        """


rule MetricIntegration:
    input:
        pre="{outdir}/pre/Gex/{dataName}/test.h5ad",  # 处理前的 AnnData 文件
        post="{outdir}/callModel/{model}/{dataname}/test.int.h5ad"  # 处理后的 AnnData 文件
    output:
        metric="{outdir}/callModel/{model}/metrics/{dataname}/int.tsv"  # 输出的评估结果文件
    log:
        "{outdir}/log/2.Int.log"
    benchmark:
        "{outdir}/benchmark/2.Int.benchmark"
    params:
        batch_key="dataset_id",  # 批次键
        label_key="cell_type",  # 标签键
        metrics_to_run=' '.join(["kBET", "graph_connectivity", "ari", "nmi", "silhouette"])  # 需要运行的指标
    shell:
        """
        python scripts/metrics/Int.py \
          -a {input.pre} \
          -b {input.post} \
          -bk {params.batch_key} \
          -lk {params.label_key} \
          -o {output.metric} \
          -bio {params.metrics_to_run} &>> {log}
        """


rule MetricRecon:
    input:
        pre="{outdir}/pre/Gex/{dataName}/test.h5ad",  # 原始数据的 h5ad 文件路径
        post="{outdir}/callModel/{model}/{dataname}/test.recon.h5ad"  # 重构后数据的 h5ad 文件路径
    output:
        metric="{outdir}/callModel/{model}/metrics/{dataname}/recon.tsv"  # 输出的评估结果文件
    log:
        "{outdir}/log/2.Recon.log"
    benchmark:
        "{outdir}/benchmark/2.Recon.benchmark"
    shell:
        """
        python scripts/metrics/Recon.py \
          -i {input.pre} \
          -p {input.post} \
          -o {output.metric} &>> {log}
        """