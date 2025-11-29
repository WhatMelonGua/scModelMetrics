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
import argparse
import os
import torch
import numpy as np
import pandas as pd
from datasets import Dataset
# from datasets import load_from_disk
from transformers import BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
# endregion

exam=['-i', '/data/run01/sczd231/disk/Projects/XD/scModelMetrics/inputs/dataset/blood_immune.h5ad', 
      't', '/data/run01/sczd231/disk/Projects/XD/scModelMetrics/inputs/dataset/blood_immune.h5ad',
      '-m', '/data/run01/sczd231/disk/Projects/XD/scModelMetrics/scripts/models/Geneformer/models/Geneformer-V1-10M',
      '-o', '/data/run01/sczd231/disk/Projects/XD/scModelMetrics/outputs/demo/callModel/geneformer',
      '-b', '32',
      '-e', '20']

# region |- Argument Parsing -|
parser = argparse.ArgumentParser(description="GeneFormer Cell Annotation Script")
parser.add_argument("-i", "--train", help="Path to the training h5ad file (required for non-zero-shot mode).")
parser.add_argument("-t", "--test", required=True, help="Path to the test h5ad file.")
parser.add_argument("-m", "--model_dir", required=True, help="Path to the trained model directory.")
parser.add_argument("-o", "--output", required=True, help="Output directory for predictions.")
# parser.add_argument("-z", "--zero_shot", action="store_true", help="Enable zero-shot mode (no training set provided).")
parser.add_argument("-b", "--batch_size", type=int, default=64, help="Batch size for training and inference (default: 64).")
parser.add_argument("-e", "--epochs", type=int, default=10, help="Number of epochs for fine-tuning (default: 3).")
args = parser.parse_args()
# endregion

# region |- Setup -|
# Set random seeds for reproducibility
seed_val = 42
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

# Ensure output directory exists
os.makedirs(args.output, exist_ok=True)
# endregion

# region |- Load Data -|
def load_data(file_path):
    """Load h5ad data and convert to dataset format."""
    import anndata as ad
    adata = ad.read_h5ad(file_path)
    data = {"input_ids": adata.X, "label": adata.obs["cell_type"].values, "cell_id": adata.obs.index.values}
    return data

def classes_to_ids(data, target_name_id_dict):
    """Convert class labels to numerical IDs."""
    data["label"] = [target_name_id_dict.get(label, 0) for label in data["label"]]
    return data

def ids_to_class(ids, id2cell_dict):
    """Convert numerical IDs back to class labels."""
    return [id2cell_dict[id] for id in ids]


# Load test data
test_data = load_data(args.test)

# Load training data if not in zero-shot mode
# if not args.zero_shot:
train_data = load_data(args.train)
# endregion

# region |- Load Model -|
max_input_size = 2 ** 12  # 4096
max_lr = 5e-5
freeze_layers = 0
num_gpus = 1
num_proc = 16
lr_schedule_fn = "linear"
warmup_steps = 500
optimizer = "adamw"
logging_steps = round(len(labeled_train_split)/geneformer_batch_size/10)



# Load the trained model
model = BertForSequenceClassification.from_pretrained(args.model_dir,
                                                    num_labels=len(target_name_id_dict.keys()),
                                                    output_attentions = False,
                                                    output_hidden_states = False)
model.cuda()

# Load label mapping (if available)
label_mapping_path = os.path.join(args.model_dir, "label_mapping.pkl")
if os.path.exists(label_mapping_path):
    import pickle
    with open(label_mapping_path, "rb") as f:
        target_name_id_dict = pickle.load(f)
    id2cell_dict = {v: k for k, v in target_name_id_dict.items()}
else:
    raise ValueError("Label mapping not found. Ensure the model directory contains 'label_mapping.pkl'.")
# endregion

# region |- Fine-tuning (if not zero-shot) -|
# if not args.zero_shot:
# Convert labels to IDs
train_data = classes_to_ids(train_data, target_name_id_dict)
# Split into training and evaluation sets
train_data, eval_data = train_test_split(train_data, test_size=0.2, random_state=seed_val)
# Convert to Hugging Face Dataset
train_dataset = Dataset.from_dict(train_data)
eval_dataset = Dataset.from_dict(eval_data)


# Define training arguments
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
    "per_device_train_batch_size": args.batch_size,
    "per_device_eval_batch_size": args.batch_size,
    "num_train_epochs": args.epochs,
    "load_best_model_at_end": True,
    "output_dir": f"{args.output}/weight"
}
training_args = TrainingArguments(
    **training_args
)
# Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=lambda pred: {
        "accuracy": accuracy_score(pred.label_ids, pred.predictions.argmax(-1)),
        "macro_f1": f1_score(pred.label_ids, pred.predictions.argmax(-1), average="macro")
    }
)
# Fine-tune the model
trainer.train()
trainer.save_model(args.output)
# endregion


# region |- Inference -|
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average='macro')
    return {"accuracy": acc, "macro_f1": macro_f1}


# Prepare test dataset
test_data = classes_to_ids(test_data, target_name_id_dict)

# Create Trainer for inference
trainer = Trainer(
    model=model,
    args=TrainingArguments(output_dir=args.output, per_device_eval_batch_size=args.batch_size),
    compute_metrics=compute_metrics
)


# Perform inference
predictions = trainer.predict(test_data)
pred_ids = predictions.predictions.argmax(axis=-1)
pred_labels = ids_to_class(pred_ids, id2cell_dict)

# Save predictions
output_file = os.path.join(args.output, "anno.tsv")
pd.DataFrame({"cell_id": test_data["cell_id"], "pred": pred_labels}, "true": test_data["label"]).to_csv(output_file, sep='\t')

print(f"Predictions saved to {output_file}")
# endregion