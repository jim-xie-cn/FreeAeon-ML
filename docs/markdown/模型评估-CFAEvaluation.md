# CFAEvaluation

## 功能分类
模型评估

## 类描述
模型评估工具类，提供分类模型评估和ROC曲线分析功能

## 应用场景

- 模型性能评估：计算准确率、精确率、召回率等指标
- 模型对比：比较不同模型的性能表现
- 阈值优化：通过ROC曲线选择最优分类阈值
- 模型诊断：识别模型的优势和不足
- 结果可视化：绘制ROC曲线展示模型效果
        

## 方法列表


### 主要方法

#### 1. evaluate(ds_pred, ds_true, average='weighted')
评估分类模型，返回多个评估指标。

#### 2. get_binary_roc(ds_true, ds_prob)
计算二分类ROC曲线数据。

#### 3. show_binary_roc(roc_auc, df_roc, title=None)
可视化显示ROC曲线。
        

## 示例代码


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from FreeAeonML.FAEvaluation import CFAEvaluation
from FreeAeonML.FASample import CFASample

# 1. 准备分类预测结果
np.random.seed(42)
n_samples = 1000
ds_true = pd.Series(np.random.randint(0, 2, n_samples))
# 模拟预测结果（添加一些噪声）
ds_pred = ds_true.copy()
noise_idx = np.random.choice(n_samples, size=int(n_samples * 0.1), replace=False)
ds_pred.iloc[noise_idx] = 1 - ds_pred.iloc[noise_idx]

# 2. 二分类评估
df_eval = CFAEvaluation.evaluate(ds_pred, ds_true, average='weighted')
print("二分类评估结果:")
print(df_eval)

# 解读指标
print("\n指标解读:")
print(f"准确率(accuracy): {df_eval['accuracy'].iloc[0]:.4f} - 预测正确的比例")
print(f"精确率(precision): {df_eval['precision'].iloc[0]:.4f} - 预测为正的样本中真正为正的比例")
print(f"召回率(recall): {df_eval['recall'].iloc[0]:.4f} - 真正为正的样本中被预测为正的比例")
print(f"F1分数(f1_score): {df_eval['f1_score'].iloc[0]:.4f} - 精确率和召回率的调和平均")
print(f"AUC: {df_eval['auc'].iloc[0]:.4f} - ROC曲线下的面积")
print(f"MCC: {df_eval['mcc'].iloc[0]:.4f} - 马修斯相关系数，范围[-1,1]")

# 混淆矩阵
print("\n混淆矩阵:")
print(df_eval['confusion_matrix'].iloc[0])

# 3. ROC曲线分析
ds_prob = pd.Series(np.random.rand(n_samples))  # 预测概率
roc_auc, df_roc = CFAEvaluation.get_binary_roc(ds_true, ds_prob)
print(f"\nROC AUC: {roc_auc:.4f}")
print("\nROC曲线数据（前5行）:")
print(df_roc.head())

# 4. 可视化ROC曲线
CFAEvaluation.show_binary_roc(roc_auc, df_roc, title="二分类模型")

# 5. 找出最优阈值（Youden指数最大）
df_roc['youden'] = df_roc['tpr'] - df_roc['fpr']
best_idx = df_roc['youden'].idxmax()
best_threshold = df_roc.loc[best_idx, 'thresholds']
best_tpr = df_roc.loc[best_idx, 'tpr']
best_fpr = df_roc.loc[best_idx, 'fpr']
print(f"\n最优阈值: {best_threshold:.4f}")
print(f"对应的TPR: {best_tpr:.4f}, FPR: {best_fpr:.4f}")

# 6. 多分类评估
ds_true_multi = pd.Series(np.random.randint(0, 3, n_samples))
ds_pred_multi = ds_true_multi.copy()
noise_idx = np.random.choice(n_samples, size=int(n_samples * 0.1), replace=False)
ds_pred_multi.iloc[noise_idx] = (ds_pred_multi.iloc[noise_idx] + 1) % 3

df_eval_multi = CFAEvaluation.evaluate(ds_pred_multi, ds_true_multi, average='weighted')
print("\n多分类评估结果:")
print(df_eval_multi[['accuracy', 'precision', 'recall', 'mcc']])

# 7. 不同平均方式对比
for avg_method in ['weighted', 'macro', 'micro']:
    df_eval_avg = CFAEvaluation.evaluate(ds_pred_multi, ds_true_multi, average=avg_method)
    print(f"\n{avg_method}平均方式:")
    print(f"  Precision: {df_eval_avg['precision'].iloc[0]:.4f}")
    print(f"  Recall: {df_eval_avg['recall'].iloc[0]:.4f}")
        

## 参数说明


| 方法 | 参数 | 类型 | 说明 |
|------|------|------|------|
| evaluate | ds_pred | pd.Series | 预测标签 |
| | ds_true | pd.Series | 真实标签 |
| | average | str | 多分类平均方式：weighted/macro/micro |
| get_binary_roc | ds_true | pd.Series | 真实标签（0或1） |
| | ds_prob | pd.Series | 预测概率 |
| show_binary_roc | roc_auc | float | AUC值 |
| | df_roc | pd.DataFrame | ROC数据 |
| | title | str | 图表标题 |
        

## 返回值说明


- **evaluate**: DataFrame包含评估指标
- **get_binary_roc**: (AUC值, ROC曲线数据)
- **show_binary_roc**: 无返回值（显示图表）
        

## 注意事项


- 评估指标包括：accuracy、precision、recall、f1_score、fbeta_score、auc、mcc
- 混淆矩阵以数组形式存储在结果中
- ROC曲线仅适用于二分类问题
- AUC值越接近1表示模型性能越好
- weighted平均考虑类别样本数，macro不考虑，micro全局计算
- 单类别问题会返回错误提示
- MCC范围[-1,1]，1表示完美预测，0表示随机预测
        

---
*生成时间: 2026-03-23 16:10:12*
*项目: FreeAeon-ML*
