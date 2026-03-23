# CFAModelCluster

## 功能分类
模型训练

## 类描述
聚类模型工具类，支持多种无监督学习算法进行数据聚类分析

## 应用场景

- 客户分群：根据客户行为将客户分为不同群体
- 图像分割：将图像像素聚类为不同区域
- 异常检测：识别不属于任何聚类的异常点
- 文档聚类：将相似文档归为一类
- 市场细分：识别不同的市场细分
        

## 方法列表


### 主要方法

#### 1. __init__(cluster_count, models=None)
初始化聚类器，指定聚类数量。

#### 2. fit_predict(df_sample)
对数据进行聚类并返回聚类结果。

#### 3. evaluate(df_sample)
评估聚类质量，返回轮廓系数、CH指数、DB指数。

#### 4. sample_cluster(df_cluster_result, df_sample, model_name)
为原始样本添加聚类标签。
        

## 示例代码


import pandas as pd
import numpy as np
from FreeAeonML.FAModelCluster import CFAModelCluster
from FreeAeonML.FASample import CFASample

# 1. 准备聚类数据
df_sample = CFASample.get_random_cluster(n_samples=500, n_features=4)
print(f"数据形状: {df_sample.shape}")

# 2. 初始化聚类模型（聚类数为3）
cluster_model = CFAModelCluster(cluster_count=3)

# 3. 执行聚类
df_cluster_result = cluster_model.fit_predict(df_sample)
print("\n聚类结果:")
print(df_cluster_result)

# 查看各模型的聚类数量
for model_name, group in df_cluster_result.groupby('model'):
    print(f"\n{model_name}模型聚类情况:")
    for _, row in group.iterrows():
        cluster_id = row['cluster']
        sample_count = len(row['rows'][0])
        print(f"  聚类{cluster_id}: {sample_count}个样本")

# 4. 评估聚类质量
df_evaluation = cluster_model.evaluate(df_sample)
print("\n聚类质量评估:")
print(df_evaluation)

# 解读评估指标
print("\n评估指标解读:")
print("轮廓系数(silhouette_score): 越接近1越好，表示聚类紧密且分离")
print("CH指数(calinski_harabasz_score): 越大越好，表示类间分散度和类内紧密度比值高")
print("DB指数(davies_bouldin_score): 越小越好，表示类内相似度高且类间差异大")

# 5. 为样本添加聚类标签
df_labeled = cluster_model.sample_cluster(df_cluster_result, df_sample, "KMeans")
print("\n带聚类标签的样本:")
print(df_labeled.head(10))
print(f"\n各聚类的样本数量:\n{df_labeled['_cluster'].value_counts()}")

# 6. 自定义聚类算法
from sklearn.cluster import KMeans, DBSCAN
custom_models = {
    "KMeans_custom": KMeans(n_clusters=3, random_state=42),
    "DBSCAN_custom": DBSCAN(eps=0.5, min_samples=5)
}
cluster_custom = CFAModelCluster(cluster_count=3, models=custom_models)
df_custom_result = cluster_custom.fit_predict(df_sample)
print("\n自定义模型聚类结果:")
print(df_custom_result)
        

## 参数说明


| 方法 | 参数 | 类型 | 说明 |
|------|------|------|------|
| __init__ | cluster_count | int | 聚类数量 |
| | models | dict | 自定义模型字典，None使用默认 |
| fit_predict | df_sample | pd.DataFrame | 待聚类数据 |
| evaluate | df_sample | pd.DataFrame | 待评估数据 |
| sample_cluster | df_cluster_result | pd.DataFrame | 聚类结果 |
| | df_sample | pd.DataFrame | 原始样本 |
| | model_name | str | 模型名称 |
        

## 返回值说明


- **fit_predict**: DataFrame包含各模型的聚类结果
- **evaluate**: DataFrame包含聚类质量评估指标
- **sample_cluster**: DataFrame包含原始数据加聚类标签
        

## 注意事项


- 默认支持8种算法：KMeans、AffinityPropagation、AgglomerativeClustering、Birch、MeanShift、OPTICS、SpectralClustering、GaussianMixture
- DBSCAN已注释，因为需要调整eps和min_samples参数
- 轮廓系数范围[-1, 1]，接近1最好
- CH指数越大越好，无上限
- DB指数越小越好，0为最优
- 聚类前建议对数据进行标准化处理
- 不同算法适用于不同数据分布
        

---
*生成时间: 2026-03-23 16:10:12*
*项目: FreeAeon-ML*
