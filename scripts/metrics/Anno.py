#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# # # # # # # # # # # # 
"""
╭───────────────────────────────────────╮ 
│ AccF1.py   2025/09/26/-16:29 
╰───────────────────────────────────────╯ 
│ Description:
    用于 单细胞注释 子任务, ACC 及 F1 评分
""" # [By: HuYw]

# region |- Import -|
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score # 评估方法
import argparse
# endregion


# 创建 argparse 解析器
parser = argparse.ArgumentParser(description="计算 准确率Acc 和 F1 分数（宏平均）")
parser.add_argument("-i", "--input", type=str, help="输入的 预测结果TSV 文件路径, 第一列是True label, 第二列是预测的 label") 
parser.add_argument("-o", "--output", type=str, help="输出结果TSV 文件路径") 
args = parser.parse_args()
# 读取 TSV 文件
try:
    data = pd.read_csv(args.input, sep="\t", header=None)
    if len(data.columns) != 2:
        raise ValueError("TSV 文件必须包含两列：第一列是真实标签，第二列是预测标签")
except Exception as e:
    print(f"读取文件时出错：{e}")


# 提取真实标签和预测标签
true_labels = data.iloc[:, 0]
predicted_labels = data.iloc[:, 1]
# 计算准确率和 F1 分数（宏平均）
accuracy = accuracy_score(true_labels, predicted_labels)
f1_macro = f1_score(true_labels, predicted_labels, average="macro")
# 输出结果
print(f"Accuracy: {accuracy*100:.2f}")
print(f"F1 macro: {f1_macro*100:.2f}")

pd.DataFrame([accuracy, f1_macro], columns=['ACC', 'F1']).to_csv(args.output, sep='\t')

