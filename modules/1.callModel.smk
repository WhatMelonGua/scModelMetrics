#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# # # # # # # # # # # # 
"""
╭───────────────────────────────────────╮ 
│ 1.callModel.smk   2025/11/28/-14:59 
╰───────────────────────────────────────╯ 
│ Description:
    模型调度 规则    
""" # [By: HuYw]

# region |- Import -|
from types import SimpleNamespace
import os
# endregion


rule Anno_GeneFormer:
    input: 
        h5ad = "{}/{dataset}.h5ad",
    output: 
        outdir = directory(),
        preh5ad = "adata.h5ad",
        obstab = "obs.tsv",
        datainfo = "Metrics.tsv",
        qc_umap = "UMAP.QC.png",
        cate_umap = "UMAP.Cate.png"
    log:
        "{outdir}/log/0.preGex.log"
    benchmark:
        "{outdir}/benchmark/0.preGex.benchmark"
    params:
        dataName = ,      # 该数据集的 名称标识
        batchKey = ,    # 批次效应来源的 key值
        moreKey = ,     # 其他要看UMAP的 obs key
        nhvg = 3000,    # 聚类使用hvg数目
        mingene = 300,  # 过滤cell
        mincell = 3,    # 过滤gene
        mtCutoff = 10,  # 线粒体表达比例 过滤标准
        species = ,     # 物种
    message:
        "Preprocess RNA modality with `scanpy` pipeline"
    shell:
        """
        {python} scripts/pre/scGex.py \
            -i {input.h5ad} \
            -o {output.outdir} \
            -BK {params.batchKey} \
            -MK {params.moreKey} \
            -n {params.nhvg} \
            -mg {} \
         &>> {log}
        """