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

configset(config, 'tool', 'configs/sets/envs.tsv')  # 工具/环境 配置
# endregion

T = configParser.Tool(config['tool'])
P = configParser.Project(config['job'])

# region |- Include .smk -|
include: "modules/0.preMatrix.smk"
include: "modules/1.callModel.smk"
include: "modules/2.doMetrics.smk"
include: "modules/3.report.smk"
# endregion


