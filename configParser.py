#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# # # # # # # # # # # # 
"""
╭───────────────────────────────────────╮ 
│ configParser.py   2025/11/28/-14:59 
╰───────────────────────────────────────╯ 
│ Description:
    Yaml 参数解析器    
""" # [By: HuYw]

# region |- Import -|
from types import SimpleNamespace
from pathlib import Path
import pandas as pd
import yaml
import os
# endregion



# region |- Env 工具类, 适用tsv配置 -|
class Env():
    def __init__(self, path):
        table = pd.read_csv(path, sep='\t', index_col=0)
        table.index = table.index.str.lower()
        self.geneformer = table.loc['geneformer', 'Path']
        self.scgpt = table.loc['scgpt', 'Path']
        self.genecompass = table.loc['genecompass', 'Path']
# endregion



# region |- Dataset 配置类 -|
class Dataset():
    def __init__(self, name, train: dict, test: dict):
        self.name = name
        self.train = SimpleNamespace(
            h5ad=Path(train['h5ad']).absolute(),
            annoKey=train['annoKey'],
            batchKey=train['batchKey'],
            species=train['species'].lower(),
        )
        self.test = SimpleNamespace(
            h5ad=Path(test['h5ad']).absolute(),
            annoKey=test['annoKey'],
            batchKey=test['batchKey'],
            species=test['species'].lower(),
        )
# endregion


# region |- 评估配置类 -|
class Metric():
    options=('recon', 'annotation', 'mod_fusion', 'integration', 'perturb', 'consume')
    def __init__(self, cfg, verbose=True):
        cfg = [c.lower() for c in cfg if c.lower() in self.options]
        if verbose:
            print(f"Support Metrics: {cfg}")
        # flag y
        self.recon = 'recon' in cfg
        self.annotation = 'annotation' in cfg
        self.mod_fusion = 'mod_fusion' in cfg
        self.integration = 'integration' in cfg
        self.perturb = 'perturb' in cfg
        self.consume = 'consume' in cfg
# endregion


# region |- Model 配置类 -|
class Model():
    def __init__(self, cfg_kv, cfg_tab, metric_set:Metric=None):
        self.name = cfg_kv['name']
        self.model_dir = Path(cfg_kv['model_dir']).absolute()
        self.script_dir = Path(f'./scripts/models/{self.name}')
        # region |- 有必要定向 可客制化 相关任务脚本路径/名称 -|
        # @todo: ...
        # endregion
        self.zeroshot = cfg_kv['zeroshot']
        # 模型功能
        # can_anno, can_recon, can_mod_int, can_batch_eff, can_perturb, can_cosume
        # annotation, recon, mod_fusion, integration, perturb, consume
        self.can_anno = cfg_tab.loc[self.name, 'Anno']
        self.can_recon = cfg_tab.loc[self.name, 'Recon']
        self.can_mod_int = cfg_tab.loc[self.name, 'ModInt']
        self.can_batch_eff = cfg_tab.loc[self.name, 'BatchEff']
        self.can_perturb = cfg_tab.loc[self.name, 'Perturb']
        self.can_cosume = cfg_tab.loc[self.name, 'Consume']
        # 按metric任务 取消评估
        if metric_set:
            self.can_anno = self.can_anno and metric_set.annotation
            self.can_recon = self.can_recon and metric_set.recon
            self.can_mod_int = self.can_mod_int and metric_set.mod_fusion
            self.can_batch_eff = self.can_batch_eff and metric_set.integration
            self.can_perturb = self.can_perturb and metric_set.perturb
            self.can_cosume = self.can_cosume and metric_set.consume
# endregion



# region |- 项目 配置类, 适用yaml配置 -|
class Task():
    def __init__(self, path):
        self.path = path
        with open(self.path, 'r') as f:
            self.yaml = yaml.safe_load(f)
        # 输出路径
        self.outdir = Path(self.yaml['Outdir']).absolute()
        self.taskRecord = self.outdir / 'superv.tsv'    # 0: 不评估, 1: 待评估, 2:已评估
        # 注册数据内容
        self.dataset = {}
        for ds in self.yaml['Dataset']:
            name, train, test = ds['name'], ds['train'], ds['test']
            self.dataset[name] = Dataset(name, train, test)
        # 加载评估配置
        self.metric = Metric(self.yaml['Metric'], verbose=True)
        # 加载模型配置
        self.model_cfg = pd.read_csv(self.yaml['ModelFuncCfg'], sep='\t', index_col=0).astype(bool)
        self.model_cfg.index = self.model_cfg.index.str.lower()
        self.model = {}
        for m in self.yaml['Model']:
            self.model[m['name']] = Model(m, self.model_cfg, self.metric)
        # 初始化评估表
        self.genMetricInfo().to_csv(self.taskRecord, sep='\t')
    # 生成模型评估记录表
    def genMetricInfo(self):
        init_record = pd.DataFrame(index=self.model.keys(), columns=self.model_cfg.columns)
        init_record.index.name = 'Model'
        for k, v in self.model:
            init_record.loc[k] = (v.can_anno, v.can_recon, v.can_mod_int, v.can_batch_eff, v.can_perturb, v.can_cosume)
        return init_record.astype(int)
    # 更新模型评估记录表
    def updateMetricInfo(self, model, task):
        record = pd.read_csv(self.taskRecord, sep='\t', index_col=0)
        record.loc[model, task] = 2
        record.to_csv(self.taskRecord, sep='\t')
    # 获取所有待 处理 数据
    
# endregion

