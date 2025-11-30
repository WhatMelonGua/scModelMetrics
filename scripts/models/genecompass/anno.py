#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# # # # # # # # # # # # 
"""
╭───────────────────────────────────────╮ 
│ train.py   2025/11/28/-13:55 
╰───────────────────────────────────────╯ 
│ Description:
    模型训练脚本 - GeneCompass    
    
""" # [By: HuYw]

# region |- Import -|
import os
import sys
import argparse
import pickle
import subprocess
import numpy as np
import random
import torch
from collections import Counter
from datasets import load_from_disk, concatenate_datasets
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import Trainer, TrainingArguments
from genecompass import BertForSequenceClassification, DataCollatorForCellClassification
from genecompass.utils import load_prior_embedding
import seaborn as sns
sns.set()
# endregion

# 参数解析
parser = argparse.ArgumentParser(description="GeneCompass Cell Annotation Script")
parser.add_argument("-i", "--train", required=True, help="Path to the training dataset (h5ad or similar format).")
parser.add_argument("-t", "--test", required=True, help="Path to the test dataset (h5ad or similar format).")
parser.add_argument("-ak", "--annoKey", required=True, help="Annotation key in the dataset's obs for cell type labels.")
parser.add_argument("-m", "--model_dir", required=True, help="Path to the pre-trained model directory.")
parser.add_argument("-o", "--output", required=True, help="Output directory for predictions and model results.")
parser.add_argument("-s", "--species", default="hg", choices=["hg", "mm"], help="Species (hg for human, mm for mouse). Default: hg.")
parser.add_argument("-bk", "--batchKey", default=None, help="Batch key in the dataset's obs (optional).")
parser.add_argument("-dn", "--dataName", default="default_dataset", help="Name of the dataset for logging purposes. Default: default_dataset.")
parser.add_argument("-b", "--batch_size", type=int, default=32, help="Batch size for training and inference. Default: 32.")
parser.add_argument("-e", "--epochs", type=int, default=20, help="Number of epochs for fine-tuning. Default: 20.")
parser.add_argument("-lr", "--learning_rate", type=float, default=5e-5, help="Learning rate for fine-tuning. Default: 5e-5.")
parser.add_argument("-f", "--freeze_layers", type=int, default=12, help="Number of layers to freeze in the model. Default: 12.")
parser.add_argument("-td", "--token_dict", default="../../prior_knowledge/human_mouse_tokens.pickle", help="Path to the token dictionary. Default: ../../prior_knowledge/human_mouse_tokens.pickle.")
args = parser.parse_args()

# 设置环境变量
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
sys.path.append("/path/to/GeneCompass/")

# 加载知识嵌入
knowledges = dict()
out = load_prior_embedding(token_dictionary_or_path=args.token_dict)
knowledges['promoter'] = out[0]
knowledges['co_exp'] = out[1]
knowledges['gene_family'] = out[2]
knowledges['peca_grn'] = out[3]
knowledges['homologous_gene_human2mouse'] = out[4]

# 加载数据集
train_set = load_from_disk(args.train)
test_set = load_from_disk(args.test)

# 重命名列
train_set = train_set.rename_column(args.annoKey, "label")
test_set = test_set.rename_column(args.annoKey, "label")

# 创建细胞类型到标签ID的字典
target_names = set(list(Counter(train_set["label"]).keys()) + list(Counter(test_set["label"]).keys()))
target_name_id_dict = dict(zip(target_names, [i for i in range(len(target_names))]))
print("Target Name to ID Mapping:", target_name_id_dict)

# 将标签转换为数值ID
def classes_to_ids(example):
    example["label"] = target_name_id_dict[example["label"]]
    return example

train_set = train_set.map(classes_to_ids, num_proc=16)
test_set = test_set.map(classes_to_ids, num_proc=16)

# 过滤测试集中的标签
trained_labels = list(Counter(train_set['label']).keys())
def if_trained_label(example):
    return example['label'] in trained_labels

test_set = test_set.filter(if_trained_label, num_proc=16)

# 定义计算指标的函数
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average="macro", zero_division=0)
    recall = recall_score(labels, preds, average="macro", zero_division=0)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'macro_f1': macro_f1
    }

# 加载预训练模型
model = BertForSequenceClassification.from_pretrained(
    args.model_dir,
    num_labels=len(target_name_id_dict.keys()),
    output_attentions=False,
    output_hidden_states=False,
    knowledges=knowledges,
)

# 冻结部分层
if args.freeze_layers > 0:
    modules_to_freeze = model.bert.encoder.layer[:args.freeze_layers]
    for module in modules_to_freeze:
        for param in module.parameters():
            param.requires_grad = False

model = model.to("cuda")

# 设置训练参数
training_args = TrainingArguments(
    output_dir=args.output,
    dataloader_num_workers=2,
    learning_rate=args.learning_rate,
    do_train=True,
    do_eval=True,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_steps=10,
    group_by_length=True,
    length_column_name="length",
    disable_tqdm=False,
    lr_scheduler_type="linear",
    warmup_steps=100,
    weight_decay=0.001,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    num_train_epochs=args.epochs,
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    greater_is_better=True,
)

# 创建Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=DataCollatorForCellClassification(),
    train_dataset=train_set,
    eval_dataset=test_set,
    compute_metrics=compute_metrics
)

# 训练模型
trainer.train()
os.makedirs(f"{args.output}/model", exist_ok=True)
trainer.save_model(f"{args.output}/model")

# 保存预测结果和模型
predictions = trainer.predict(test_set)
with open(f"{args.output}/predictions.pickle", "wb") as fp:
    pickle.dump(predictions, fp)


trainer.save_metrics("eval", predictions.metrics)