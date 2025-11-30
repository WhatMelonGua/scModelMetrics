#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# # # # # # # # # # # # 
"""
╭───────────────────────────────────────╮ 
│ test.py   2025/11/28/-13:55 
╰───────────────────────────────────────╯ 
│ Description:
    模型测试脚本 - GeneFormer
""" # [By: HuYw]


# region |- Import -|
import scanpy as sc
from geneformer import TranscriptomeTokenizer, EmbExtractor
import argparse
import os
import torch
import numpy as np
import pandas as pd
from collections import Counter
import pandas as pd
import numpy as np
import math
import datetime
import pickle
import subprocess
import seaborn as sns
from datasets import load_from_disk
from transformers import BertForSequenceClassification
from transformers import Trainer, DataCollatorWithPadding
from transformers.training_args import TrainingArguments
from geneformer import DataCollatorForCellClassification
from sklearn.metrics import accuracy_score, f1_score
import gc
# import wandb
# run = wandb.init(
#     project="GeneFormer",
#     reinit=True,
#     settings=wandb.Settings(start_method="fork"),
#     mode="offline", # wandb
# )

import time
import random
import torch
# 设置所有seed
seed_num = 0
random.seed(seed_num)
np.random.seed(seed_num)
seed_val = 42
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)
# endregion

exam=['-i', '/data/run01/sczd231/disk/Projects/XD/scModelMetrics/outputs/demo/pre/Gex/train.h5ad', 
      '-t', '/data/run01/sczd231/disk/Projects/XD/scModelMetrics/outputs/demo/pre/Gex/test.h5ad',
      '-s', 'hg',
      '-ak', 'cell_type',
      '-bk', 'dataset_id',
      '-dn', 'blood_immune',
      '-m', '/data/run01/sczd231/disk/Projects/XD/scModelMetrics/scripts/models/geneformer/models/Geneformer-V1-10M',
      '-o', '/data/run01/sczd231/disk/Projects/XD/scModelMetrics/outputs/demo/callModel/geneformer',
      '-b', '32',
      '-e', '20',
      '-gs', '/data/run01/sczd231/disk/Projects/XD/scModelMetrics/scripts/models/geneformer/models/geneSet.csv']

# region |- Argument Parsing -|
parser = argparse.ArgumentParser(description="GeneFormer Cell Annotation Script")
parser.add_argument("-i", "--train", help="Path to the training h5ad file (required for non-zero-shot mode).")
parser.add_argument("-t", "--test", required=True, help="Path to the test h5ad file.")
parser.add_argument('-s', '--species', default='hg', help='species')
parser.add_argument("-ak", "--annoKey", required=True, help="obs anno key.")
parser.add_argument("-bk", "--batchKey", required=True, help="obs batch key.")
parser.add_argument("-dn", "--dataName", required=True, help="dataname.")
parser.add_argument("-m", "--model_dir", required=True, help="Path to the trained model directory.")
parser.add_argument("-o", "--output", required=True, help="Output directory for predictions.")
# parser.add_argument("-z", "--zero_shot", action="store_true", help="Enable zero-shot mode (no training set provided).")
parser.add_argument("-b", "--batch_size", type=int, default=64, help="Batch size for training and inference (default: 64).")
parser.add_argument("-e", "--epochs", type=int, default=10, help="Number of epochs for fine-tuning (default: 3).")
parser.add_argument("-gs", "--geneset", required=True, help="geneset table.")
args = parser.parse_args()
# endregion

# Ensure output directory exists
os.makedirs(args.output, exist_ok=True)

model_version='V1' if 'V1' in os.path.basename(args.model_dir) else 'V2'
tk = TranscriptomeTokenizer({args.annoKey: "label", }, nproc=8, model_version=model_version)
# region |- make data transfer -|
train=sc.read(args.train)
test=sc.read(args.test)

geneSet=pd.read_csv(args.geneset, index_col=0)
geneSet.index = geneSet['gene_name'].copy()
def update_h5advar(adata, geneset, species='hg'):
    n_vars = adata.n_vars
    # n_counts
    qc = sc.pp.calculate_qc_metrics(adata, inplace=False)
    adata.obs['n_counts']=qc[0]['total_counts'].values
    if not adata.X.max().is_integer():
        adata.X = adata.layers['counts']
    if species == 'mm':
        print("Mouse upper gene name")
        adata.var_names = [x.upper() for x in adata.var_names]
        adata = adata[:, adata.var_names.intersection(geneSet['gene_name'])]
        var_names = geneSet.loc[adata.var_names, 'ensembl_id']
        var_names = var_names[~var_names.index.duplicated(keep='first')]    # 删除重复项！
        adata.var_names = var_names.values
        adata.var['ensembl_id'] = geneSet.loc[adata.var_names, 'ensembl_id'].duplicated(keep='first')
    else:
        ensg_mapped = set(geneSet['ensembl_id']).intersection(adata.var['gene_ids'])
        adata = adata[:, adata.var['gene_ids'].isin(ensg_mapped)]
        adata.var['ensembl_id'] = adata.var['gene_ids'].copy()
    print(f"keep gene: {n_vars} > {adata.n_vars}")
    return adata


train=update_h5advar(train, geneSet, args.species)
test=update_h5advar(test, geneSet, args.species)


os.makedirs(f"{args.output}/dataset/H5train", exist_ok=True)
train.write(f"{args.output}/dataset/H5train/train.h5ad")
os.makedirs(f"{args.output}/dataset/H5test", exist_ok=True)
test.write(f"{args.output}/dataset/H5test/test.h5ad")

print("Tokenizing ...")
tk.tokenize_data(f"{args.output}/dataset/H5train", 
                 f"{args.output}/dataset/",
                 "train",
                 file_format="h5ad")

tk.tokenize_data(f"{args.output}/dataset/H5test", 
                 f"{args.output}/dataset/",
                 "test",
                 file_format="h5ad")

del train, test
gc.collect()


# endregion


# region |- Load Data -|

#train_dataset=load_from_disk('/xtdisk/jiangl_group/wangqifei/hu/ANNO/comp-pipeline/methods/GeneFormer/path/to/cell_type_train_data.dataset')
train_dataset=load_from_disk(f"{args.output}/dataset/train.dataset")
target_names = list(Counter(train_dataset["label"]).keys())
target_name_id_dict = dict(zip(target_names,[i for i in range(len(target_names))]))


def trainSplit(total,rate,order=None):
    cont = round(total*rate)
    mark = np.array([True]*total)
    if order is None:
        mark[np.random.choice(range(mark.shape[0]), size=cont, replace=False)]=False
        return mark
    if (order+1 < 1/rate):
        base = round(order*total*rate)
        mark[base:(base+cont)]=False
    else:
        mark[(total-cont):(total+1)]=False
    return mark


# change labels to numerical ids
def classes_to_ids(example):
    example["label"] = target_name_id_dict.get(example["label"], 0)
    return example


def ids_to_class(ids):
    return [id2cell_dict[id] for id in ids]


id2cell_dict = dict(zip(target_name_id_dict.values(), target_name_id_dict.keys()))
labeled_trainset = train_dataset.map(classes_to_ids, num_proc=16)

index_data = np.arange(0,labeled_trainset.num_rows)
train_mask = trainSplit(labeled_trainset.num_rows, 0.2)

index_train = index_data[train_mask]
index_eval = index_data[~train_mask]

# create 80/20 train/eval splits
labeled_eval_split = labeled_trainset.select(index_eval)
labeled_train_split = labeled_trainset.select(index_train)
# endregion


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    # calculate accuracy and macro f1 using sklearn's function
    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average='macro')
    return {
        'accuracy': acc,
        'macro_f1': macro_f1
    }


# region |- Train model -|
# set model parameters
max_input_size = 2 ** 12  # 4096
# set training hyperparameters
max_lr = 5e-5
freeze_layers = 0
num_gpus = 1
num_proc = 16
geneformer_batch_size = args.batch_size
lr_schedule_fn = "linear"
warmup_steps = 500
epochs = 10
optimizer = "adamw"


# set logging steps
logging_steps = round(len(labeled_train_split)/geneformer_batch_size/10)


# reload pretrained model
model = BertForSequenceClassification.from_pretrained(args.model_dir, 
                                                    num_labels=len(target_name_id_dict.keys()),
                                                    output_attentions = False,
                                                    output_hidden_states = False,
                                                    )

model.cuda()
# define output directory path
current_date = datetime.datetime.now()
process_dir = f"{args.output}/weight"
output_dir = args.output

# ensure not overwriting previously saved model
saved_model_test = os.path.join(process_dir, f"pytorch_model.bin")
if os.path.isfile(saved_model_test) == True:
    raise Exception("Model already saved to this directory.")

# make output directory
subprocess.call(f'mkdir -p {process_dir}', shell=True)
subprocess.call(f'mkdir -p {output_dir}', shell=True)


# Define training arguments
training_args = {
    "learning_rate": max_lr,
    "do_train": True,
    "do_eval": True,
    "eval_strategy": "epoch",
    "save_strategy": "epoch",
    "logging_steps": logging_steps,
    "group_by_length": True,
    "length_column_name": "length",
    "disable_tqdm": False,
    "lr_scheduler_type": lr_schedule_fn,
    "warmup_steps": warmup_steps,
    "weight_decay": 0.001,
    "per_device_train_batch_size": args.batch_size,
    "per_device_eval_batch_size": args.batch_size,
    "num_train_epochs": args.epochs,
    "load_best_model_at_end": True,
    "output_dir": process_dir
}
training_args = TrainingArguments(
    **training_args
)
# Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=DataCollatorForCellClassification(token_dictionary = {
    # 示例键值对
    "input_ids": "input_ids",
    "label": "label",
    "length": "length"
    }),
    train_dataset=labeled_train_split,
    eval_dataset=labeled_eval_split,
    compute_metrics=compute_metrics
)
# Fine-tune the model
trainer.train()
trainer.save_model(process_dir)


# region |- Pred by Model -|
test_dataset=load_from_disk(f"{args.output}/dataset/test.dataset")
target_labels = test_dataset['label']   # !
# ids"label" map to code
labeled_testset = test_dataset.map(classes_to_ids, num_proc=16)
target_ids = labeled_testset['label']
# region |- Anno -|

predictions = trainer.predict(labeled_testset)
pred_ids = predictions.predictions.argmax(axis=-1)
pred_labels = ids_to_class(pred_ids)   # !

anno = pd.DataFrame([target_labels, pred_labels], index=['True', 'Pred']).T
anno.to_csv(f"{output_dir}/anno.tsv", sep='\t')

# endregion

# region |- Integration -|

# initiate EmbExtractor
# OF NOTE: model_version should match version of model to be used (V1 or V2) to use the correct token dictionary
embex = EmbExtractor(model_type="CellClassifier",
                     num_classes=len(target_names),
                     max_ncells=None,
                     emb_layer=0,
                     emb_label=['label'],
                     labels_to_plot=['label'],
                     forward_batch_size=200,
                     model_version="V1",  # OF NOTE: SET TO V1 MODEL, PROVIDE V1 MODEL PATH IN SUBSEQUENT CODE
                     nproc=8)

# extracts embedding from input data
# input data is tokenized rank value encodings generated by Geneformer tokenizer (see tokenizing_scRNAseq_data.ipynb)
# example dataset for V1 model series: https://huggingface.co/datasets/ctheodoris/Genecorpus-30M/tree/main/example_input_files/cell_classification/disease_classification/human_dcm_hcm_nf.dataset
embs = embex.extract_embs(process_dir, # example V1 fine-tuned model
                          f"{args.output}/dataset/test.dataset",
                          f"{args.output}/emb",
                          "test")
# int adata output:
test=sc.read(args.test)
test.obsm['X_int'] = embs.values
test.write(f"{output_dir}/test.int.h5ad")
# endregion
