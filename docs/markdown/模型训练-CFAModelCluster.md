# CFAModelCluster - 聚类模型

## 应用场景

CFAModelCluster类提供多种聚类算法,主要应用于:

- 客户分群和市场细分
- 异常检测
- 数据探索和模式发现
- 图像分割
- 推荐系统

## 安装依赖

```bash
pip install FreeAeon-ML
```

## 支持的聚类算法

- **KMeans**: K均值聚类
- **AffinityPropagation**: 亲和力传播
- **AgglomerativeClustering**: 层次聚类
- **Birch**: BIRCH算法
- **MeanShift**: 均值漂移
- **OPTICS**: OPTICS密度聚类
- **SpectralClustering**: 谱聚类
- **GaussianMixture**: 高斯混合模型

## 初始化参数

| 参数名 | 类型 | 说明 |
|--------|------|------|
| cluster_count | int | 聚类数量 |
| models | dict | 自定义模型字典(可选) |

## 方法详解

### 1. fit_predict - 聚类预测

```python
def fit_predict(self, df_sample)
```

对数据进行聚类并返回每个样本的聚类标签。

**示例**:
```python
from FreeAeonML.FAModelCluster import CFAModelCluster
from FreeAeonML.FASample import CFASample

df_sample = CFASample.get_random_cluster()
cluster_model = CFAModelCluster(cluster_count=3)
df_result = cluster_model.fit_predict(df_sample)
print(df_result)
```

### 2. evaluate - 聚类评估

```python
def evaluate(self, df_sample)
```

计算聚类质量指标:
- silhouette_score: 轮廓系数(越大越好)
- calinski_harabasz_score: CH指数(越大越好)
- davies_bouldin_score: DB指数(越小越好)

**示例**:
```python
df_perf = cluster_model.evaluate(df_sample)
print(df_perf)
```

### 3. sample_cluster - 提取聚类结果

```python
def sample_cluster(self, df_cluster_result, df_sample, model_name)
```

将聚类标签添加到原始数据。

**示例**:
```python
df_clustered = cluster_model.sample_cluster(df_result, df_sample, "KMeans")
print(df_clustered.head())
```

## 完整示例

```python
import matplotlib.pyplot as plt
from FreeAeonML.FAModelCluster import CFAModelCluster
from FreeAeonML.FASample import CFASample

# 生成数据
df_sample = CFASample.get_random_cluster()

# 聚类
cluster_model = CFAModelCluster(cluster_count=3)
df_result = cluster_model.fit_predict(df_sample)

# 评估
df_perf = cluster_model.evaluate(df_sample)
print("聚类性能:")
print(df_perf.sort_values('silhouette_score', ascending=False))

# 可视化最佳模型
best_model = df_perf.loc[df_perf['silhouette_score'].idxmax(), 'model_name']
df_clustered = cluster_model.sample_cluster(df_result, df_sample, best_model)

plt.figure(figsize=(10, 6))
for cluster in df_clustered['_cluster'].unique():
    subset = df_clustered[df_clustered['_cluster'] == cluster]
    plt.scatter(subset.iloc[:, 0], subset.iloc[:, 1], label=f'Cluster {cluster}')
plt.legend()
plt.title(f'{best_model} Clustering')
plt.show()
```

## 相关类链接

- [CFAFeatureSelect](./特征工程-CFAFeatureSelect.md) - 特征降维
