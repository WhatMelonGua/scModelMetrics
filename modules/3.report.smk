#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# # # # # # # # # # # # 
"""
╭───────────────────────────────────────╮ 
│ 3.report.smk   2025/11/28/-14:59 
╰───────────────────────────────────────╯ 
│ Description:
    汇报总结 规则    
""" # [By: HuYw]

# region |- Import -|
from types import SimpleNamespace
import os
# endregion

# 设计开发中, 将输出的Metric聚合在单个文件夹下
rule AggMetrics:
    input:
        int_tsv="{outdir}/callModel/{model}/metrics/{dataname}/anno.tsv"  # 注释评估结果文件
        int_tsv="{outdir}/callModel/{model}/metrics/{dataname}/int.tsv"  # 整合评估结果文件
        int_tsv="{outdir}/callModel/{model}/metrics/{dataname}/recon.tsv"  # 重构评估结果文件
    output:
        "{outdir}/Metrics/{dataname}/{model}.tsv"  # 输出目录
    shell:
        """
        cat {input.anno_tsv}  {input.recon_tsv}  {input.int_tsv} \
        >> {output}
        """