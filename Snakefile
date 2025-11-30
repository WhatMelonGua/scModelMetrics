#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# # # # # # # # # # # # 
"""
╭───────────────────────────────────────╮ 
│ Snakefile   2025/11/28/-14:58 
╰───────────────────────────────────────╯ 
│ Description:
    模型评估流程    
""" # [By: HuYw]

# region |- Import -|
from types import SimpleNamespace
import configParser
import os
# endregion


# region |- Config -|
def configset(config:dict, k:str, v):
    if k not in config:
        config[k] = v

# 设置snakemake默认配置项 k=v
for k in ('job',):
    assert k in config.keys(), \
    f"Error: No config '{k}' input, please use:\n'snakemake --config job=configs/demo.yaml -j 2' to start a job!"

configset(config, 'env', 'configs/sets/envs.tsv')  # 工具/环境 配置
# endregion

E = configParser.Tool(config['env'])    # 环境配置加载
T = configParser.Task(config['job'])    # 任务计划加载

# region |- Include .smk 加载子模块 -|
include: "modules/0.preMatrix.smk"
include: "modules/1.callModel.smk"
include: "modules/2.doMetrics.smk"
include: "modules/3.report.smk"
# endregion

# 根据子模块规则 定制 -> 全流程执行至生成报告路径 [metrics开发不全, 暂时计划优化]
# 按 数据集 打组输出报告
rule summary_all:
    input:
        expand("{outdir}/{dataset}/report/summary.tsv", outdir=T.outdir, dataset=T.dataset.keys()),
    message:
        "[Main] Whole Report Pipeline start..."


# 根据子模块规则 定制 -> 全流程执行至生成报告路径 [metrics开发不全, 暂时计划优化]
# 按 数据集 打组输出报告
rule all:
    input:
        expand("{outdir}/{dataset}/report/summary.tsv", outdir=T.outdir, dataset=T.dataset.keys()),
    message:
        "[Main] Whole Report Pipeline start..."


