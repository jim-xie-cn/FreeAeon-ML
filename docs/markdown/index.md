# FreeAeon-ML 使用文档索引

FreeAeon-ML是一个机器学习工具库，提供数据探索、预处理、特征工程、模型训练、评估和可视化的完整解决方案。

---

## 快速开始

### 安装

```bash
pip install FreeAeon-ML
```

### 基本依赖

```bash
pip install pandas numpy scikit-learn h2o matplotlib seaborn scipy statsmodels imbalanced-learn pyvis pyecharts
```

---

## 功能分类与文档导航

### 1. 数据工具 (Data Utility)

#### [CFASample - 样本生成与数据处理](数据工具-CFASample.md)

**应用场景**：生成测试数据、数据集划分、不平衡数据处理

**主要方法**：

- `get_random_regression()` - 生成回归样本
- `get_random_classification()` - 生成分类样本
- `get_random_cluster()` - 生成聚类样本
- `split_dataset()` - 划分训练测试集
- `resample_smote()` - SMOTE重采样处理不平衡数据
- `resample_balance()` - 简单平衡重采样
- `find_changed_index()` - 查找时序变化点
- `split_dataframe()` - 批量划分数据

---

### 2. 数据探索 (Exploratory Data Analysis)

#### [CFADataDistribution - 数据分布分析](数据探索-CFADataDistribution.md)

**应用场景**：分布可视化、正态性检验、分布对比

**主要方法**：

- `show_dist()` - 显示分布图
    - 直方图 (Histogram)
    - 核密度估计 (KDE)
    - Q-Q图
- `normal_test()` - 正态性检验
    - Shapiro-Wilk检验
    - Kolmogorov-Smirnov检验
    - DAgostino-Pearson检验
- `dist_test()` - 分布相似性检验
    - F检验
    - 卡方检验
- `normal_fit()` - 正态分布拟合

#### [CFADataTest - 假设检验](数据探索-CFADataTest.md)

**应用场景**：时间序列平稳性检验、因果关系分析

**主要方法**：

- `stationarity_test()` - 平稳性检验 (ADF检验)
- `show_acf_pacf()` - 显示自相关和偏自相关图
- `granger_test()` - 格兰杰因果检验

#### [CFAFitter - 数据拟合](数据探索-CFAFitter.md)

**应用场景**：曲线拟合、趋势分析

**主要方法**：

- `fit()` - 数据拟合，支持三种模式：
    - linear - 线性拟合
    - polynomial - 多项式拟合
    - exponential - 指数拟合
- `plot()` - 绘制拟合曲线
- `get_fit_data()` - 获取拟合数据
- `get_fit_param()` - 获取拟合参数

#### [CFACommonStats - 统计量计算](数据探索-CFACommonStats.md)

**应用场景**：描述性统计分析、数据特征提取

**主要方法**：

- `get_one_stats()` - 计算单变量统计量
- `get_stats()` - 计算多变量统计量
    - 基本统计: mean, std, cv
    - 分布特征: skew, kurt_fisher, kurt_pearson
    - 自相关: autocorr_lag1
    - 范围统计: min, max, range, iqr

---

### 3. 数据预处理 (Data Preprocessing)

#### [CFADataPreprocess - 数据预处理](数据预处理-CFADataPreprocess.md)

**应用场景**：数据转换、异常值处理、标准化

**主要方法**：

- `normal_transform()` - Box-Cox正态转换
- `normal_recovery()` - 逆正态转换
- `polyfit()` - 多项式拟合
- `get_abnormal()` - 检测异常值 (Z-Score方法)
- `remove_abnormal()` - 移除异常值
- `get_scale()` - 标准化处理
    - z-score - Z-score标准化
    - max-min - Min-Max标准化
- `get_transformer_position_encoding()` - Transformer位置编码
- `assign_qbins()` - 等频率分箱

#### [CFATransformer - 数据分布转换](数据预处理-CFATransformer.md)

**应用场景**：分布映射、域适应、数据增强

**主要方法**：

- `fit()` - 学习源分布到目标分布的映射
- `transform()` - 转换数据分布
- `inverse()` - 逆转换恢复原始分布

**支持模式**：

- scale - 线性缩放（不改变分布）
- quantile - 分位数映射（单变量/多变量）
- copula - Copula变换（多变量，支持相关性调整）

---

### 4. 特征工程 (Feature Engineering)

#### [CFAFeatureSelect - 特征选择](特征工程-CFAFeatureSelect.md)

**应用场景**：特征筛选、降维、因果分析

**主要方法**：

- `load()` - 加载数据到H2O框架
- `get_inform_graph()` - 生成信息图 (基于H2O)
    - 支持算法: AUTO, deeplearning, drf, gbm, glm, xgboost
    - 显示特征的总信息量和净信息量
- `get_anovaglm()` - GLM-ANOVA方差检验
- `granger_test()` - 格兰杰因果检验
- `get_data_pca()` - PCA和t-SNE降维可视化
- `get_matrix_by_pca()` - PCA矩阵降维和重构

---

### 5. 模型训练 (Model Training)

#### [CFAModelClassify - 分类模型](模型训练-CFAModelClassify.md)

**应用场景**：自动化分类模型训练（支持二分类和多分类）

**主要方法**：

- `train()` - 训练多个分类模型
- `save()` - 保存模型到文件
- `load()` - 从文件加载模型
- `predict()` - 预测类别和概率
- `evaluate()` - 评估模型性能
    - Accuracy - 准确率
    - Precision - 精确率
    - Recall - 召回率
    - F1-Score - F1分数
    - AUC - ROC曲线下面积
    - MCC - Matthews相关系数
- `importance()` - 特征重要性分析

**支持算法**：

- DT - 决策树
- SVM - 支持向量机
- RF - 随机森林
- ANN - 人工神经网络
- Bayes - 朴素贝叶斯
- GLM - 广义线性模型
- GBM - 梯度提升机
- XGBoost - 极端梯度提升（平台相关）

#### [CFAModelRegression - 回归模型](模型训练-CFAModelRegression.md)

**应用场景**：自动化回归模型训练

**主要方法**：

- `train()` - 训练多个回归模型
- `save()` - 保存模型
- `load()` - 加载模型
- `predict()` - 预测数值
- `evaluate()` - 评估模型性能
    - MSE - 均方误差
    - RMSE - 均方根误差
    - MAE - 平均绝对误差
    - R² - 决定系数
- `importance()` - 特征重要性分析

**支持算法**：

- RF - 随机森林
- ANN - 人工神经网络
- GLM - 广义线性模型
- GBM - 梯度提升机
- XGBoost - 极端梯度提升（平台相关）

#### [CFAModelCluster - 聚类模型](模型训练-CFAModelCluster.md)

**应用场景**：无监督聚类分析

**主要方法**：

- `fit_predict()` - 训练并预测聚类标签
- `evaluate()` - 评估聚类性能
- `sample_cluster()` - 为样本添加聚类标签

**支持算法**：

- KMeans - K均值聚类
- AffinityPropagation - 亲和传播
- AgglomerativeClustering - 层次聚类
- Birch - BIRCH聚类
- MeanShift - 均值漂移
- OPTICS - OPTICS聚类
- SpectralClustering - 谱聚类
- GaussianMixture - 高斯混合模型

**评估指标**：

- Silhouette Score - 轮廓系数
- Calinski-Harabasz Score - Calinski-Harabasz指数
- Davies-Bouldin Score - Davies-Bouldin指数

#### [CFAArima - 时间序列模型](模型训练-CFAArima.md)

**应用场景**：时间序列预测、季节性分析

**主要方法**：

- `fit()` - ARIMA模型拟合
- `predict()` - 预测训练期数据
- `forecast()` - 预测未来数据
- `auto_fit()` - 自动搜索最佳参数
- `decomposition()` - 季节性分解
    - 趋势分量
    - 季节分量
    - 残差分量
- `show_model()` - 显示模型摘要信息
- `show_result()` - 可视化预测结果

---

### 6. 模型评估 (Model Evaluation)

#### [CFAEvaluation & CFASimilarity - 评估与相似度](模型评估-CFAEvaluation.md)

**CFAEvaluation - 模型评估**

**应用场景**：分类模型性能评估、ROC曲线分析

**主要方法**：

- `evaluate()` - 分类模型综合评估
    - 混淆矩阵
    - Recall - 召回率
    - MCC - Matthews相关系数
    - Accuracy - 准确率
    - Precision - 精确率
    - F1-Score - F1分数
    - AUC - ROC曲线下面积
- `get_binary_roc()` - 计算ROC曲线数据
- `show_binary_roc()` - 绘制ROC曲线

**CFASimilarity - 相似度计算**

**应用场景**：序列相似度比较、分布一致性检验

**主要方法**：

- `Cosine()` - 余弦相似度
- `Pearson()` - 皮尔森相关系数
- `Euclidean()` - 欧氏距离
- `Manhattan()` - 曼哈顿距离
- `Minkowski()` - 闵可夫斯基距离
- `Jaccard()` - Jaccard相似度系数
- `KSTest()` - Kolmogorov-Smirnov检验
- `EMD()` - 地球移动距离

---

### 7. 数据可视化 (Visualization)

#### [CFAVisualize - 通用可视化](可视化-CFAVisualize.md)

**应用场景**：数据可视化、统计图表

**主要方法**：

- `show_heatmap()` - 热力图展示
    - 适用于相关性矩阵
    - 数据表格可视化
- `get_sankey()` - 桑基图（流向图）
    - 支持单DataFrame或列表
    - 自动计算节点和流量
- `show_contour()` - 等高线图
    - 支持黑白或彩色
    - 可选数值标注
- `show_sequence()` - 时序数据网络图

#### [CFANetGraph - 网络图可视化](可视化-CFANetGraph.md)

**应用场景**：关系网络、流程图、时序可视化

**主要方法**：

- `Add_Node()` - 添加节点
    - 自定义标签、颜色、大小
    - 支持分组
    - 悬停提示
- `Add_Edge()` - 添加边
    - 设置权重
    - 添加标签
- `Add()` - 便捷添加节点和边
- `Show()` - 生成交互式HTML图形
    - 支持拖拽
    - 支持缩放
    - 交互式浏览
- `Show_Series()` - 时序数据可视化
    - 自动连接节点
    - 自定义边信息
- `call_edge_info()` - 边信息计算

**特性**：

- 支持有向图和无向图
- 交互式HTML输出
- 自动浏览器打开

---

### 8. 通用工具 (Common Utilities)

#### [CFACommon - 通用功能](通用工具-CFACommon.md)

**应用场景**：数据IO、通用工具

**主要方法**：

- `load_csv()` - 带进度条的CSV加载
    - 自动显示加载进度
    - 支持大文件分块读取

---

## 典型工作流程

### 工作流程1：分类任务完整流程

```python
# 步骤1: 生成或加载数据
from FreeAeonML.FASample import CFASample
df_data = CFASample.get_random_classification(
    n_sample=1000,
    n_feature=10,
    n_class=2
)

# 步骤2: 数据探索
from FreeAeonML.FADataEDA import CFADataDistribution, CFACommonStats

# 统计分析
stats = CFACommonStats.get_stats(df_data)
print(stats)

# 分布检验
result = CFADataDistribution.normal_test(df_data['x0'])
print(result)

# 步骤3: 数据预处理
from FreeAeonML.FADataPreprocess import CFADataPreprocess

# 标准化
df_scaled, _ = CFADataPreprocess.get_scale(
    df_data,
    y_column=['y'],
    scale_type='z-score'
)

# 步骤4: 特征选择
from FreeAeonML.FAFeatureSelect import CFAFeatureSelect
import h2o

h2o.init(nthreads=-1, verbose=False)

selector = CFAFeatureSelect()
selector.load(df_scaled, y_column='y')
ig = selector.get_inform_graph("AUTO")

# 步骤5: 数据集划分
df_train, df_test = CFASample.split_dataset(
    df_scaled,
    test_ratio=0.2
)

# 步骤6: 模型训练
from FreeAeonML.FAModelClassify import CFAModelClassify

model = CFAModelClassify()
model.train(df_train, y_column='y', train_ratio=0.85)

# 步骤7: 模型评估
df_evaluate = model.evaluate(df_test, y_column='y')
print(df_evaluate)

# 步骤8: 特征重要性
df_importance = model.importance()
print(df_importance)
```

### 工作流程2：时间序列分析

```python
# 步骤1: 加载数据
import pandas as pd
ds_data = pd.Series([...])  # 你的时序数据

# 步骤2: 平稳性检验
from FreeAeonML.FADataEDA import CFADataTest

result = CFADataTest.stationarity_test(ds_data)
print(result)

# 步骤3: 查看ACF/PACF
CFADataTest.show_acf_pacf(ds_data)

# 步骤4: ARIMA建模
from FreeAeonML.FAModelSeries import CFAArima
from FreeAeonML.FASample import CFASample

df_train, df_test = CFASample.split_dataset(
    pd.DataFrame({'y': ds_data}),
    test_ratio=0.2
)

model, order = CFAArima.auto_fit(
    ds_train=df_train['y'],
    ds_test=df_test['y'],
    seasonal_order_range=((1,3), (0,2), (1,3), (1,12))
)

# 步骤5: 预测未来
forecast = CFAArima.forecast(model, num_future_points=10)
print(forecast)

# 步骤6: 可视化
CFAArima.show_result(df_test['y'], model.predict())
```

### 工作流程3：聚类分析

```python
# 步骤1: 生成数据
from FreeAeonML.FASample import CFASample

df_data = CFASample.get_random_cluster(n_sample=500)

# 步骤2: 降维可视化
from FreeAeonML.FAFeatureSelect import CFAFeatureSelect

df_pca, df_tsne = CFAFeatureSelect.get_data_pca(
    df_samples=df_data,
    n_components=2,
    label_column='y',
    with_plot=True
)

# 步骤3: 聚类
from FreeAeonML.FAModelCluster import CFAModelCluster

cluster_model = CFAModelCluster(cluster_count=2)
df_result = cluster_model.fit_predict(df_data[['x1', 'x2']])

# 步骤4: 评估
df_perf = cluster_model.evaluate(df_data[['x1', 'x2']])
print(df_perf)

# 步骤5: 获取聚类标签
df_clustered = cluster_model.sample_cluster(
    df_result,
    df_data,
    model_name="KMeans"
)
print(df_clustered)
```

---

## 注意事项

### H2O初始化

使用分类/回归/特征选择模块前需要初始化H2O：

```python
import h2o
h2o.init(nthreads=-1, verbose=False)
```

**WSL环境**：使用 `h2o.connect()` 替代 `h2o.init()`

### 数据格式

- 大多数方法接受 pandas DataFrame 或 Series
- 目标列在分类任务中会自动转换为因子类型

### 内存管理

- 处理大数据集时注意内存占用
- H2O会占用独立的JVM内存

### 模型保存

- 分类和回归模型支持保存和加载
- 聚类和ARIMA模型不支持持久化

---

## 平台兼容性

| 模块 | Linux | macOS (Intel) | macOS (ARM) | Windows |
|------|-------|---------------|-------------|---------|
| 基础功能 | ✅ | ✅ | ✅ | ✅ |
| H2O (非XGBoost) | ✅ | ✅ | ✅ | ✅ |
| XGBoost | ✅ | ✅ | ❌ | ❌ |

---

## 版本信息

- **文档版本**: 1.0
- **生成日期**: 2026-03-24
- **源代码仓库**: https://github.com/jim-xie-cn/FreeAeon-ML
- **功能概述**: https://github.com/jim-xie-cn/FreeAeon-ML/issues/6

---

## 相关资源

- **GitHub仓库**: https://github.com/jim-xie-cn/FreeAeon-ML
- **问题反馈**: 请在GitHub仓库提交Issue
- **文档首页**: 从本文档开始浏览

---
