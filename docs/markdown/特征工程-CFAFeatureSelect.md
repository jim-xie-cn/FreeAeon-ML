# CFAFeatureSelect

## 功能分类
特征工程

## 类描述
特征选择工具类，提供信息图分析、方差检验、因果检验等特征筛选方法

## 应用场景

- 分类问题特征选择：使用信息图识别重要特征
- 回归问题特征分析：使用GLM-ANOVA检验特征显著性
- 时序因果分析：使用格兰杰因果检验分析变量间因果关系
- 降维可视化：使用PCA和t-SNE降维并可视化
- 特征重要性排序：识别对模型最有价值的特征
        

## 方法列表


### 主要方法

#### 1. load(df_sample, x_columns=[], y_column='label', is_regression=False)
加载数据并准备特征选择。

#### 2. get_inform_graph(algorithm="AUTO", protected_columns=[])
生成信息图，分析特征的预测能力和独特性。

#### 3. get_anovaglm(family='gaussian', lambda_=0, highest_interaction_term=2)
使用GLM-ANOVA进行特征显著性检验。

#### 4. granger_test(ds_result, ds_source, maxlag, p_value=0.05)
格兰杰因果检验，分析时序数据的因果关系。

#### 5. get_data_pca(df_samples, n_components=2, label_column='y', with_plot=True)
使用PCA和t-SNE降维并可视化。

#### 6. get_matrix_by_pca(dataMat, n)
使用PCA降维并重构矩阵。
        

## 示例代码


import h2o
import pandas as pd
import numpy as np
from FreeAeonML.FAFeatureSelect import CFAFeatureSelect
from FreeAeonML.FASample import CFASample

# 初始化H2O
h2o.init(nthreads=-1, verbose=False)

# 1. 分类问题的信息图分析
df_sample = CFASample.get_random_classification(1000, n_feature=10, n_class=2)
fea = CFAFeatureSelect()
fea.load(df_sample, y_column='y', is_regression=False)

# 生成信息图（单一算法）
ig = fea.get_inform_graph(algorithm="AUTO")
ig.plot()  # 显示信息图
admissible_features = ig.get_admissible_features()
print("可接受的特征:", admissible_features)

# 生成多算法信息图
ig_all = fea.get_inform_graph(algorithm="All")
for algor, ig_obj in ig_all.items():
    print(f"{algor}算法的可接受特征:", ig_obj.get_admissible_features())

# 2. 回归问题的方差检验
df_regression = CFASample.get_random_regression(1000)
fea.load(df_regression, x_columns=['x1', 'x2'], y_column='y', is_regression=True)
anova_model = fea.get_anovaglm(family='gaussian', highest_interaction_term=2)
print(anova_model.summary())

# 3. 格兰杰因果检验
ds_cause = pd.Series([1, -1, 2, -2, 3, -3, 4, -4])
ds_effect = pd.Series([2, -2, 3, -3, 4, -4, 5, -5])
min_p, best_lag, approve_list, detail = CFAFeatureSelect.granger_test(
    ds_result=ds_effect,
    ds_source=ds_cause,
    maxlag=3,
    p_value=0.05
)
print(f"最佳滞后阶数: {best_lag}, 最小p值: {min_p}")
print(f"通过检验的滞后阶数: {approve_list}")

# 4. PCA和t-SNE降维可视化
df_data = pd.DataFrame(np.random.rand(500, 10))
df_data['y'] = np.random.randint(0, 3, 500)
df_pca, df_tsne = CFAFeatureSelect.get_data_pca(
    df_data,
    n_components=2,
    label_column='y',
    with_plot=True
)
print("PCA降维结果:\n", df_pca.head())
print("t-SNE降维结果:\n", df_tsne.head())

# 5. 矩阵PCA降维
data_matrix = np.random.rand(100, 20)
low_dim_data, recon_data = CFAFeatureSelect.get_matrix_by_pca(data_matrix, n=5)
print(f"降维后形状: {low_dim_data.shape}")
print(f"重构后形状: {recon_data.shape}")
        

## 参数说明


| 方法 | 参数 | 类型 | 说明 |
|------|------|------|------|
| load | df_sample | pd.DataFrame | 训练数据 |
| | x_columns | list | 特征列，默认自动检测 |
| | y_column | str | 目标列名 |
| | is_regression | bool | 是否为回归问题 |
| get_inform_graph | algorithm | str | 算法：AUTO/All/deeplearning/drf/gbm/glm |
| | protected_columns | list | 保护列（不参与选择） |
| get_anovaglm | family | str | 分布族：gaussian/binomial/poisson等 |
| | lambda_ | float | 正则化参数 |
| | highest_interaction_term | int | 最高交互项阶数 |
| granger_test | ds_result | pd.Series | 结果序列（平稳） |
| | ds_source | pd.Series | 原因序列（平稳） |
| | maxlag | int/list | 最大滞后阶数 |
| | p_value | float | 显著性水平 |
| get_data_pca | df_samples | pd.DataFrame | 待降维数据 |
| | n_components | int | 降维维度（2或3） |
| | label_column | str | 标签列 |
| | with_plot | bool | 是否可视化 |
| | perplexity | int | t-SNE困惑度 |
| | n_clusters | int | 聚类数（无标签时） |
        

## 返回值说明


- **get_inform_graph**: H2OInfogram对象或字典（algorithm='All'时）
- **get_anovaglm**: H2OANOVAGLMEstimator模型对象
- **granger_test**: (最小p值, 最佳lag, 通过检验的lag列表, 详细结果)
- **get_data_pca**: (PCA降维数据, t-SNE降维数据)
- **get_matrix_by_pca**: (降维矩阵, 重构矩阵)
        

## 注意事项


- 信息图中位于虚线右上方的特征是最佳特征
- ANOVA检验的p值<0.05表示特征显著
- 格兰杰检验要求时序数据必须平稳
- PCA保留方差信息，t-SNE保留局部结构
- 使用前需要初始化H2O环境
- 降维可能损失部分信息
        

---
*生成时间: 2026-03-23 16:10:12*
*项目: FreeAeon-ML*
