#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# # # # # # # # # # # # 
"""
╭───────────────────────────────────────╮ 
│ 0.preMatrix.smk   2025/11/28/-14:33 
╰───────────────────────────────────────╯ 
│ Description:
    数据预处理 规则
""" # [By: HuYw]

# region |- Import -|
from types import SimpleNamespace
import os
# endregion

# 获取预处理输入文件列表
def genGexInputs():


# 规划预处理输出文件列表
def genGexOutputs():


rule preGex:
    # 据 bcf 确定donor 性别 [3.step 并行]
    input: 
        train = lambda wildcards: T[wildcards.dataName].train.h5ad,
        test = lambda wildcards: T[wildcards.dataName].test.h5ad,
    output: 
        outdir = directory(f"{T.outdir}/Gex/{dataName}"),
        preI = f"{T.outdir}/Gex/{dataName}/train.h5ad",
        preT = f"{T.outdir}/Gex/{dataName}/test.h5ad",
        preI = f"{T.outdir}/Gex/{dataName}/train.Metrics.tsv",
        preT = f"{T.outdir}/Gex/{dataName}/test.Metrics.tsv",
        umapI = f"{T.outdir}/Gex/{dataName}/train.UMAP.Cate.png",
        umapT = f"{T.outdir}/Gex/{dataName}/test.UMAP.Cate.png",
    log:
        "{outdir}/log/0.preGex.log"
    benchmark:
        "{outdir}/benchmark/0.preGex.benchmark"
    params:
        dataName = lambda wildcards: wildcards.dataName,      # 该数据集的 名称标识
        moreKeyI = lambda wildcards: T[wildcards.dataName].train.annoKey,     # 其他要看UMAP的 obs key
        moreKeyT = lambda wildcards: T[wildcards.dataName].test.annoKey,     # 其他要看UMAP的 obs key
        batchKeyI = lambda wildcards: T[wildcards.dataName].train.batchKey,    # 批次效应来源的 key值
        batchKeyT = lambda wildcards: T[wildcards.dataName].test.batchKey,    # 批次效应来源的 key值
        speciesI = lambda wildcards: T[wildcards.dataName].train.species,     # 物种
        speciesT = lambda wildcards: T[wildcards.dataName].test.species,     # 物种
        nhvg = 3000,    # 聚类使用hvg数目
        mingene = 300,  # 过滤cell
        mincell = 3,    # 过滤gene
        mtCutoff = 10,  # 线粒体表达比例 过滤标准
    message:
        "Preprocess RNA modality with `scanpy` pipeline"
    shell:
        """
        {python} scripts/pre/scGex.py \
            -i {input.train} \
            -o {output.outdir}/train \
            -BK {params.batchKeyI} \
            -MK {params.moreKeyI} \
            -sp {params.speciesI} \
            -n {params.nhvg} \
            -mg {params.mingene} \
            -mc {params.mincell} \
            -mt {params.mtCutoff} \
         &>> {log} &
        {python} scripts/pre/scGex.py \
            -BK {params.batchKeyT} \
            -MK {params.moreKeyT} \
            -sp {params.speciesT} \
            -n {params.nhvg} \
            -mg {params.mingene} \
            -mc {params.mincell} \
            -mt {params.mtCutoff} \
         &>> {log} &
        wait
        """