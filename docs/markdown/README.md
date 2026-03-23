# FreeAeon-ML 使用文档

> 机器学习工具库 - 提供数据探索、预处理、特征工程、模型训练和评估的完整解决方案

**生成时间:** 2026-03-23 16:10:12

## 📚 目录


### 数据工具

- [CFACommon](数据工具-CFACommon.md) - 通用工具类，提供带进度条的CSV文件读取功能
- [CFASample](数据工具-CFASample.md) - 样本数据生成和处理工具类，提供生成测试数据、数据集划分、数据重采样等功能

### 数据探索分析

- [CFADataDistribution](数据探索分析-CFADataDistribution.md) - 数据分布分析类，提供分布可视化、正态性检验、分布比较等功能
- [CFADataTest](数据探索分析-CFADataTest.md) - 数据假设检验类，提供平稳性检验、格兰杰因果检验、自相关分析等功能
- [CFAFitter](数据探索分析-CFAFitter.md) - 数据拟合类，支持线性拟合、多项式拟合和指数拟合
- [CFACommonStats](数据探索分析-CFACommonStats.md) - 统计量计算类，提供全面的描述性统计分析功能

### 数据预处理

- [CFADataPreprocess](数据预处理-CFADataPreprocess.md) - 数据预处理工具类，提供数据清洗、标准化、异常检测等预处理功能
- [CFATransformer](数据预处理-CFATransformer.md) - 数据风格转换工具，支持将一组数据的分布特征转换为另一组数据的分布特征

### 特征工程

- [CFAFeatureSelect](特征工程-CFAFeatureSelect.md) - 特征选择工具类，提供信息图分析、方差检验、因果检验等特征筛选方法

### 模型训练

- [CFAModelClassify](模型训练-CFAModelClassify.md) - 自动化分类模型训练工具，支持多种分类算法的批量训练、评估和预测
- [CFAModelRegression](模型训练-CFAModelRegression.md) - 自动化回归模型训练工具，支持多种回归算法的批量训练、评估和预测
- [CFAModelCluster](模型训练-CFAModelCluster.md) - 聚类模型工具类，支持多种无监督学习算法进行数据聚类分析
- [CFAArima](模型训练-CFAArima.md) - ARIMA时间序列分析工具，提供模型拟合、预测和自动参数选择功能

### 模型评估

- [CFAEvaluation](模型评估-CFAEvaluation.md) - 模型评估工具类，提供分类模型评估和ROC曲线分析功能
- [CFASimilarity](模型评估-CFASimilarity.md) - 相似度计算工具类，提供多种距离和相似度度量方法

### 可视化

- [CFAVisualize](可视化-CFAVisualize.md) - 数据可视化工具类，提供热图、桑基图、等高线图等多种可视化方法
- [CFANetGraph](可视化-CFANetGraph.md) - 网络图可视化工具类，用于创建和展示复杂的节点-边关系网络

## 🚀 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 基本使用

```python
from FreeAeonML.FASample import CFASample
from FreeAeonML.FAModelClassify import CFAModelClassify

# 生成样本数据
df_sample = CFASample.get_random_classification(1000, n_feature=10, n_class=2)

# 训练模型
model = CFAModelClassify()
model.train(df_sample, y_column='y')
```

## 📖 文档说明

本文档包含了FreeAeon-ML库中所有类的详细使用说明，每个类的文档包括：

- **功能描述**: 类的主要功能和用途
- **应用场景**: 适用的实际应用场景
- **方法列表**: 所有可用方法及其说明
- **示例代码**: 完整的使用示例
- **参数说明**: 详细的参数说明
- **返回值说明**: 方法返回值的详细说明
- **注意事项**: 使用时需要注意的事项

## 🔗 相关链接

- [GitHub仓库](https://github.com/jim-xie-cn/FreeAeon-ML)
- [问题反馈](https://github.com/jim-xie-cn/FreeAeon-ML/issues)

---
*本文档由自动化工具生成*
