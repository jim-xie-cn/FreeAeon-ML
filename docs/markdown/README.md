# FreeAeon-ML 使用文档索引

## 项目简介

FreeAeon-ML是一个基于Python的机器学习工具库,提供从数据探索、预处理、特征工程到模型训练和评估的完整机器学习流程。

项目地址: https://github.com/jim-xie-cn/FreeAeon-ML

功能概述: https://github.com/jim-xie-cn/FreeAeon-ML/issues/6

## 安装

```bash
pip install FreeAeon-ML
```

## 文档分类目录

### 1. 数据探索

数据探索阶段用于了解数据的分布特征、统计特性和时间序列特性。

- [CFADataTest - 数据假设检验](./数据探索-CFADataTest.md)
    - 平稳性检验(ADF Test)
    - ACF/PACF自相关分析
    - 格兰杰因果检验

- [CFAFitter - 数据拟合](./数据探索-CFAFitter.md)
    - 线性拟合
    - 多项式拟合
    - 指数拟合

- [CFACommonStats - 统计量计算](./数据探索-CFACommonStats.md)
    - 集中趋势(均值、中位数、众数)
    - 离散程度(标准差、方差、变异系数)
    - 分布形态(偏度、峰度)
    - 自相关系数

- [CFADataDistribution - 分布检验](./数据探索-CFADataDistribution.md)
    - 分布可视化(直方图、KDE、Q-Q图)
    - 正态性检验
    - 分布相似性检验

### 2. 数据预处理

数据预处理阶段用于清洗、转换和标准化数据。

- [CFADataPreprocess - 基础预处理](./数据预处理-CFADataPreprocess.md)
    - Box-Cox正态转换
    - 异常值检测与移除(Z-Score)
    - 多项式拟合
    - 特征标准化(Z-Score/Min-Max)
    - 等频分箱
    - Transformer位置编码

- [CFATransformer - 高级数据转换](./数据预处理-CFATransformer.md)
    - 线性缩放(scale模式)
    - 分位数映射(quantile模式)
    - Copula变换(copula模式)
    - 分布转换与域适应

### 3. 特征工程

特征工程阶段用于选择、提取和创建有效特征。

- [CFAFeatureSelect - 特征选择](./特征工程-CFAFeatureSelect.md)
    - Infogram信息图分析(分类)
    - ANOVA方差分析(回归)
    - 格兰杰因果检验(时序)
    - PCA降维
    - t-SNE可视化

### 4. 模型训练

模型训练阶段提供多种机器学习算法的自动化训练。

- [CFAModelClassify - 分类模型](./模型训练-CFAModelClassify.md)
    - 决策树(DT)
    - 支持向量机(SVM)
    - 随机森林(RF)
    - 神经网络(ANN)
    - 朴素贝叶斯
    - GLM/GBM/XGBoost

- [CFAModelRegression - 回归模型](./模型训练-CFAModelRegression.md)
    - 随机森林回归
    - 神经网络回归
    - GLM回归
    - GBM回归
    - XGBoost回归

- [CFAModelCluster - 聚类模型](./模型训练-CFAModelCluster.md)
    - K-Means
    - 层次聚类
    - DBSCAN/OPTICS
    - 谱聚类
    - 高斯混合模型

- [CFAArima - 时间序列模型](./模型训练-CFAArima.md)
    - ARIMA建模
    - 参数自动优化
    - 时间序列分解
    - 预测与回测

### 5. 模型评估

模型评估阶段用于评价模型性能和计算相似度。

- [CFAEvaluation - 模型评估与相似度](./模型评估-CFAEvaluation.md)
    - 分类指标(准确率、精确率、召回率、F1、AUC)
    - ROC曲线绘制
    - 序列相似度(余弦、皮尔森、欧氏距离等)
    - 分布检验(KS Test)

### 6. 可视化

可视化阶段提供多种数据和结果的可视化方法。

- [CFAVisualize - 数据可视化](./可视化-CFAVisualize.md)
    - 热力图
    - 桑基图(流向图)
    - 等高线图
    - 时间序列可视化
    - 网络关系图

### 7. 通用工具

通用工具提供辅助功能。

- [CFACommon - 通用工具类](./通用工具-CFACommon.md)
    - 大文件加载(带进度条)

- [CFASample - 样本数据生成](./数据工具-CFASample.md)
    - 分类样本生成
    - 回归样本生成
    - 聚类样本生成
    - 数据集划分

## 快速开始示例

### 分类任务完整流程

```python
import h2o
from FreeAeonML.FASample import CFASample
from FreeAeonML.FADataPreprocess import CFADataPreprocess
from FreeAeonML.FAFeatureSelect import CFAFeatureSelect
from FreeAeonML.FAModelClassify import CFAModelClassify

# 1. 初始化H2O
h2o.init(nthreads=-1, verbose=False)

# 2. 生成/加载数据
df_data = CFASample.get_random_classification(1000, n_feature=10, n_class=2)
df_train, df_test = CFASample.split_dataset(df_data)

# 3. 数据预处理
df_train_scaled, _ = CFADataPreprocess.get_scale(df_train, y_column=['y'])

# 4. 特征选择
fs = CFAFeatureSelect()
fs.load(df_train_scaled, y_column='y', is_regression=False)
ig = fs.get_inform_graph(algorithm='AUTO')
selected_features = ig.get_admissible_features()

# 5. 模型训练
model = CFAModelClassify()
model.train(df_train_scaled[selected_features + ['y']], y_column='y')

# 6. 模型评估
df_eval = model.evaluate(df_test, y_column='y')
print(df_eval)

# 7. 保存模型
model.save('./my_classifier')
```

### 回归任务完整流程

```python
import h2o
from FreeAeonML.FASample import CFASample
from FreeAeonML.FAModelRegression import CFAModelRegression

h2o.init(nthreads=-1, verbose=False)

# 生成数据
df_data = CFASample.get_random_regression(1000)
df_train, df_test = CFASample.split_dataset(df_data)

# 训练
model = CFAModelRegression()
model.train(df_train, y_column='y')

# 评估
df_eval = model.evaluate(df_test, y_column='y')
print(df_eval.sort_values('rmse'))
```

### 时间序列预测流程

```python
import pandas as pd
import numpy as np
from FreeAeonML.FADataEDA import CFADataTest
from FreeAeonML.FAModelSeries import CFAArima

# 生成时间序列
data = pd.Series(np.random.randn(200).cumsum())

# 平稳性检验
stationarity = CFADataTest.stationarity_test(data)
print(stationarity)

# 划分训练测试集
train, test = data[:160], data[160:]

# 自动优化ARIMA参数
model, params = CFAArima.auto_fit(train, test)

# 预测未来
forecast = CFAArima.forecast(model, num_future_points=20)
print(forecast)
```

## 常见问题

### 1. H2O相关

**Q: 如何在WSL中使用H2O?**

A: 使用 `h2o.connect()` 而非 `h2o.init()`:
```python
h2o.connect(ip='localhost', port=54321)
```

**Q: XGBoost不可用?**

A: XGBoost在Windows和ARM架构上不支持,使用其他算法替代。

### 2. 数据处理

**Q: Box-Cox转换报错?**

A: Box-Cox仅支持正值,需先处理负值:
```python
data_shifted = data - data.min() + 1
transformed, lambda_val = CFADataPreprocess.normal_transform(data_shifted)
```

**Q: 如何处理缺失值?**

A: 建议在预处理阶段处理:
```python
df_clean = df.dropna()  # 删除
# 或
df_filled = df.fillna(df.mean())  # 填充
```

### 3. 模型训练

**Q: 模型训练很慢?**

A: 可以减少训练数据或调整train_ratio:
```python
model.train(df_sample[:1000], train_ratio=0)  # 不划分验证集
```

**Q: 特征重要性为空?**

A: 某些模型不支持特征重要性,如朴素贝叶斯。

## 技术支持

- GitHub Issues: https://github.com/jim-xie-cn/FreeAeon-ML/issues
- 项目文档: https://github.com/jim-xie-cn/FreeAeon-ML

## 版本说明

当前文档版本: 1.0
文档生成日期: 2026-03-24

## 贡献者

- jim_xie - 项目作者

---

*注: 如有疑问请参考源代码或提交Issue。*
