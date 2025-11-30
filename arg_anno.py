# coding:utf-8

import argparse
import os
import sys
import time
#  +------------------+
#  | 导入自定义工具   |
#  +------------------+
sys.path.append( os.path.join('/data/run01/scw6c99/disk/BenchMark/_tool/') )
import pys.tool as tools  
import pys.metric as pysmetric
tools.setMethod('GeneFormer')  

cmd = ['-i', 'Dataset/ms_brain.sub.train.dataset', '-t', 'Dataset/ms_brain.sub.test.dataset', '-p', 'Logger/ms_brain.sub', '-k', '0', '-o', 'Logger/ms_brain.sub', '-b', '160']
cmd = ['-i', 'Dataset/ms_brain.super.train.dataset', '-t', 'Dataset/ms_brain.super.test.dataset', '-p', 'Logger/ms_brain.super', '-k', '0', '-o', 'Logger/ms_brain.super', '-b', '64']

#  +------------+
#  |  参数解析  |
#  +------------+
parser = argparse.ArgumentParser(description='Train for CellTypist cell annotation')
parser.add_argument('-i', '--input', required=True, help='Input path of dataset file as training set', type=str)
parser.add_argument('-t', '--test', required=True, help='Input path of dataset file as test set', type=str)
parser.add_argument('-p', '--process', required=True, help='Directory to save process output', type=str)
parser.add_argument('-k', '--kfold', required=False, default=0, help='K', type=int)
parser.add_argument('-o', '--output', required=True, help='Directory to save predict annotation validation.csv output', type=str)
parser.add_argument('-b', '--batch', required=False, default=64, help='Batch Size', type=int)
args = parser.parse_args()

"""
class Arg():
    def __init__(self):
        pass

args = Arg()
args.input = '../data/train/mPancreas.h5ad'
args.test = '../data/test/mPancreas.h5ad'
args.process = f'./Logger/{args.input}/'
"""

kfold = args.kfold

import os
#GPU_NUMBER = [0]
#os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(s) for s in GPU_NUMBER])
os.environ["NCCL_DEBUG"] = "INFO"
# imports
from collections import Counter
import pandas as pd
import numpy as np
import math
import datetime
import pickle
import subprocess
import seaborn as sns; sns.set()
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, f1_score
from transformers import BertForSequenceClassification
from transformers import Trainer
from transformers.training_args import TrainingArguments

from geneformer import DataCollatorForCellClassification

from geneformer import DataCollatorForCellClassification
import wandb
run = wandb.init(
    project="GeneFormer",
    reinit=True,
    settings=wandb.Settings(start_method="fork"),
    mode="offline", # wandb
)


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


"""
Run Please
"""
def roundInt(x):
    return( math.floor(x+0.5) )


def trainSplit(total,rate,order):
    cont = roundInt(total*rate)
    mark = np.array([True]*total)
    if (order+1 < 1/rate):
        base = roundInt(order*total*rate)
        mark[base:(base+cont)]=False
    else:
        mark[(total-cont):(total+1)]=False
    return mark


#train_dataset=load_from_disk('/xtdisk/jiangl_group/wangqifei/hu/ANNO/comp-pipeline/methods/GeneFormer/path/to/cell_type_train_data.dataset')
train_dataset=load_from_disk(args.input)
# train_dataset=train_dataset.rename_column(args.labelkey,'label')
# create dictionary of cell types : label ids
target_names = list(Counter(train_dataset["label"]).keys())
target_name_id_dict = dict(zip(target_names,[i for i in range(len(target_names))]))



# --------------------
#  2048 最大tensor长度
# --------------------
# input_ids columns set in 2048, the max tensor for this model!
#model_readLength=min(2048,max(train_dataset['length']))
#def std_ids(cell):
#    cell["input_ids"] = cell["input_ids"][0:model_readLength]
#    cell["length"] = model_readLength
#    return cell

# change labels to numerical ids
def classes_to_ids(example):
    example["label"] = target_name_id_dict.get(example["label"], 0)
    return example


id2cell_dict = dict(zip(target_name_id_dict.values(), target_name_id_dict.keys()))
def ids_to_class(ids):
    return [id2cell_dict[id] for id in ids]


#labeled_trainset = train_dataset.map(std_ids, num_proc=16)
labeled_trainset = train_dataset.map(classes_to_ids, num_proc=16)

index_data = np.arange(0,labeled_trainset.num_rows)
train_mask = trainSplit(labeled_trainset.num_rows,0.2, kfold)

index_train = index_data[train_mask]
index_eval = index_data[~train_mask]

# create 80/20 train/eval splits
labeled_eval_split = labeled_trainset.select(index_eval)
labeled_train_split = labeled_trainset.select(index_train)


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

# set model parameters
# max input size
max_input_size = 2 ** 12  # 4096

# set training hyperparameters
# max learning rate
max_lr = 5e-5
# how many pretrained layers to freeze
freeze_layers = 0
# number gpus
num_gpus = 1
# number cpu cores
num_proc = 16
# batch size for training and eval
geneformer_batch_size = args.batch
# learning schedule
lr_schedule_fn = "linear"
# warmup steps
warmup_steps = 500
# number of epochs
epochs = 10
# optimizer
optimizer = "adamw"


# set logging steps
logging_steps = round(len(labeled_train_split)/geneformer_batch_size/10)


# reload pretrained model
model = BertForSequenceClassification.from_pretrained("./Model/", 
                                                    num_labels=len(target_name_id_dict.keys()),
                                                    output_attentions = False,
                                                    output_hidden_states = False)

model.cuda()

# define output directory path
current_date = datetime.datetime.now()
process_dir = args.process
output_dir = args.output

# ensure not overwriting previously saved model
saved_model_test = os.path.join(process_dir, f"pytorch_model.bin")
if os.path.isfile(saved_model_test) == True:
    raise Exception("Model already saved to this directory.")

# make output directory
try:
    subprocess.call(f'mkdir {process_dir}', shell=True)
except:
    print('dict existed! process',process_dir)


try:
    subprocess.call(f'mkdir {output_dir}', shell=True)
except:
    print('dict existed! output',output_dir)

# set training arguments
training_args = {
    "learning_rate": max_lr,
    "do_train": True,
    "do_eval": True,
    "evaluation_strategy": "epoch",
    "save_strategy": "epoch",
    "logging_steps": logging_steps,
    "group_by_length": True,
    "length_column_name": "length",
    "disable_tqdm": False,
    "lr_scheduler_type": lr_schedule_fn,
    "warmup_steps": warmup_steps,
    "weight_decay": 0.001,
    "per_device_train_batch_size": geneformer_batch_size,
    "per_device_eval_batch_size": geneformer_batch_size,
    "num_train_epochs": epochs,
    "load_best_model_at_end": True,
    "output_dir": process_dir
}

training_args_init = TrainingArguments(**training_args)

# create the trainer
trainer = Trainer(
    model=model,
    args=training_args_init,
    data_collator=DataCollatorForCellClassification(),
    train_dataset=labeled_train_split,
    eval_dataset=labeled_eval_split,
    compute_metrics=compute_metrics
)
print('Train-------------------\n\n\n')

toolTracker = tools.track_on()
# train the cell type classifier
trainer.train()
toolTrackerTwo = tools.track_off(toolTracker)
print(f"Train Time Use: {toolTrackerTwo} s | from {toolTracker} s")

print('Pred Val-------------------\n\n\n')
predictions = trainer.predict(labeled_eval_split)

with open(f"{output_dir}/valid.pickle", "wb") as fp:
    pickle.dump(predictions, fp)

trainer.save_metrics("eval",predictions.metrics)
trainer.save_model(process_dir)

# store label_ids
with open(f"{output_dir}/label_ids.pickle", "wb") as fp:
    pickle.dump(target_name_id_dict, fp)


print('Pred Test-------------------\n\n\n')
test_dataset=load_from_disk(args.test)
target_labels = test_dataset['label']   # !

toolTracker = tools.track_on()
# ids"label" map to code
labeled_testset = test_dataset.map(classes_to_ids, num_proc=16)
target_ids = labeled_testset['label']
# pred also
predictions = trainer.predict(labeled_testset)
toolTrackerTwo = tools.track_off(toolTracker)
pred_ids = predictions.predictions.argmax(axis=-1)
pred_labels = ids_to_class(pred_ids)   # !

tools.store_label(list(range(len(target_labels))), list(target_labels), list(pred_labels),  f"{output_dir}/labels.csv")

with open(f"{output_dir}/test.pickle", "wb") as fp:
    pickle.dump(predictions, fp)

trainer.save_metrics("test", predictions.metrics)
print('----------TIME-------------')
print(f"Test Time Use: {toolTrackerTwo} s | from {toolTracker} s")

print('\ntest Metric ---------------------->')
print(compute_metrics(predictions))

def compute_metrics2(targ, pred):
    acc = accuracy_score(targ, pred)
    macro_f1 = f1_score(targ, pred, average='macro')
    return {
        'accuracy': acc,
        'macro_f1': macro_f1
    }

mets = pysmetric.score(target_labels, pred_labels, index='GeneFormer')
mets.to_csv(f'{output_dir}/realtest_metric.csv')
print('\ntest Metric ---------------------->')
print(mets)


