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




rule Anno_Integrate_GeneFormer:
    input:
        train = "{outdir}/pre/Gex/{dataName}/train.h5ad"
        test = "{outdir}/pre/Gex/{dataName}/test.h5ad"
    output:
        outdir=directory("{outdir}/callModel/geneformer/{dataname}") ,
        annoTab="{outdir}/callModel/geneformer/{dataname}/anno.tsv",
        intH5="{outdir}/callModel/geneformer/{dataname}/test.int.h5ad",
    log:
        anno="{outdir}/log/1.GeneFormer.Anno.log"
        integ="{outdir}/log/1.GeneFormer.Int.log"
    benchmark:
        "{outdir}/benchmark/1.GeneFormer.benchmark"
    params:
        species="hg",
        annoKey="cell_type",
        batchKey="dataset_id",
        dataName="blood_immune",
        model_dir="/data/run01/sczd231/disk/Projects/XD/scModelMetrics/scripts/models/geneformer/models/Geneformer-V1-10M",
        output_dir="/data/run01/sczd231/disk/Projects/XD/scModelMetrics/outputs/demo/callModel/geneformer",
        batch_size=32,
        epochs=20
        geneset="/data/run01/sczd231/disk/Projects/XD/scModelMetrics/scripts/models/geneformer/models/geneSet.csv"
    shell:
        """
        python scripts/models/geneformer/anno.py \
          -i {input.train_h5ad} \
          -t {input.test_h5ad} \
          -s {params.species} \
          -ak {params.annoKey} \
          -bk {params.batchKey} \
          -dn {params.dataName} \
          -m {params.model_dir} \
          -o {params.output_dir} \
          -b {params.batch_size} \
          -e {params.epochs} \
          -gs {params.geneset} &>> {log.anno}
          ln -s {log.anno} {log.integ} # 二合一脚本
        """


rule Anno_scGPT:
    input:
        train = "{outdir}/pre/Gex/{dataName}/train.h5ad"
        test = "{outdir}/pre/Gex/{dataName}/test.h5ad"
    output:
        outdir=directory("{outdir}/callModel/scgpt/{dataName}") ,
        annoTab="{outdir}/callModel/scgpt/{dataname}/test.int.h5ad",
        annoH5="{outdir}/callModel/scgpt/{dataname}/test.anno.h5ad",
    log:
        "{outdir}/log/1.scGPT.Anno.log"
    benchmark:
        "{outdir}/benchmark/1.scGPT.Anno.benchmark"
    params:
        species=lambda wildcards: T[wildcards.dataName].test.species,
        annoKey=lambda wildcards: T[wildcards.dataName].test.annoKey,
        batchKey=lambda wildcards: T[wildcards.dataName].test.batchKey,
        dataName=lambda wildcards: wildcards.dataName,
        model_dir='./scripts/models/geneformer/gits/Geneformer/Geneformer-V1-10M',
        batch_size=32,
        epochs=20
    shell:
        """
        python scripts/models/scgpt/anno.py \
          -i {input.train_h5ad} \
          -t {input.test_h5ad} \
          -s {params.species} \
          -ak {params.annoKey} \
          -bk {params.batchKey} \
          -dn {params.dataName} \
          -m {params.model_dir} \
          -o {output.outdir} \
          -b {params.batch_size} \
          -e {params.epochs} \
          -gs {input.geneset_csv} &>> {log}
        """

rule Integrate_scGPT:
    input:
        train = "{outdir}/pre/Gex/{dataName}/train.h5ad"
        test = "{outdir}/pre/Gex/{dataName}/test.h5ad"
    output:
        outdir=directory("{outdir}/callModel/scgpt/{dataname}") ,
        intH5="{outdir}/callModel/scgpt/{dataname}/test.int.h5ad",
    log:
        "{outdir}/log/1.scGPT.Int.log"
    benchmark:
        "{outdir}/benchmark/1.scGPT.Int.benchmark"
    params:
        species=lambda wildcards: T[wildcards.dataName].test.species,
        annoKey=lambda wildcards: T[wildcards.dataName].test.annoKey,
        batchKey=lambda wildcards: T[wildcards.dataName].test.batchKey,
        dataName=lambda wildcards: wildcards.dataName,
        mouse=lambda wildcards: 'mm' in T[wildcards.dataName].test.species.lower(),
        model_dir="./scripts/models/scgpt/models/scGPT_human",
        batch_size=256,
        zero=False,
        continuous=False,
        lr=1e-4,
        nhvg=3000
    shell:
        """
        python scripts/models/scgpt/integrate.py \
          -i {input.train} \
          -t {input.test} \
          -ak {params.annoKey} \
          -bk {params.batchKey} \
          -m {params.model_dir} \
          -o {output.outdir} \
          -b {params.batch_size} \
          {"" if not params.mouse else "--mouse"} \
          {"" if not params.zero else "--zero"} \
          {"" if not params.continuous else "--continuous"} \
          -l {params.lr} \
          -n {params.nhvg} &>> {log}
        """