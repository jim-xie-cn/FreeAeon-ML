# FreeAeon-ML

**FreeAeon-ML** 是一个一站式的 Python 机器学习工具包，封装了常用的机器学习流程模块，包括数据探索分析、数据预处理、特征选择、模型训练（分类、回归、聚类、时间序列）、模型评估和可视化，旨在帮助研究者和工程师高效构建、训练和评估机器学习模型。

---

## 🚀 特性功能

- 📊 **数据探索与统计分析**：正态性检验、分布拟合、相关性分析等
- 🧹 **数据预处理**：标准化、异常值处理、Box-Cox 变换、分箱等
- 🔍 **特征选择**：信息图谱、方差分析、PCA 降维、Granger 因果检验等
- 🧠 **模型训练支持**：
  - 分类模型：DT, RF, SVM, ANN, GLM, Naive Bayes, GBM, XGBoosting,...
  - 回归模型：RF, ANN, GLM, GBM, XGBoosting,...
  - 聚类模型：GaussianMixture,KMeans,AffinityPropagation,AgglomerativeClustering,Birch,MeanShift,OPTICS,...
  - 时间序列模型：ARIMA分解与预测等
- 📈 **模型评估**：评估指标自动输出、特征重要性排序、ROC等曲线绘制
- 💾 **模型保存与加载**
- 🧬 **样本均衡与增强**：SMOTE平衡采样、经典采样、自动切分等
- 📊 **可视化支持**：热力图、等高线、桑基图、序列图等
- ⚙️ **H2O 引擎集成**：支持GPU，支持分布式，支持多客户端并发等

---

## 📦 安装方式

```bash
pip install FreeAeon-ML
```

### ✅ 环境依赖

- Python >= 3.8
- Java Runtime Environment (JRE) 8+
- 主要依赖库：
  - numpy, pandas, matplotlib, seaborn
  - scipy, scikit-learn, statsmodels
  - h2o

> 📌 **注意：必须安装 Java 环境！**
> FreeAeon-ML 使用 H2O 平台进行部分模型训练，需确保系统已安装 Java：

```bash
java -version
```

若未安装，请参考以下方式：

- macOS: `brew install java`
- Ubuntu: `sudo apt install default-jre`
- Windows: [Oracle Java 下载地址](https://www.oracle.com/java/technologies/javase-downloads.html)

---

## 🧪 快速示例

```python
import numpy as np
import pandas as pd
from FreeAeonML.FADataPreprocess import CFADataPreprocess
from FreeAeonML.FASample import CFASample
from FreeAeonML.FAModelClassify import CFAModelClassify
from h2o.estimators import H2ORandomForestEstimator
import h2o

#初始化
h2o.init()

# 随机生成样本（有5个特征，2个分类，分类标签字段为"y")
df_sample = CFASample.get_random_classification(1000, n_feature=5, n_class=2)
print(df_sample)

# 划分为训练集和测试集（默认80%为训练样本，20%为测试样本)
df_train, df_test = CFASample.split_dataset(df_sample)

# 使用自带的一组模型进行训练
model = CFAModelClassify(models=None)

# 如需要指定的模型进行训练，请按照以下格式指定模型
#model = CFAModelClassify(models={"rf": H2ORandomForestEstimator()})

# 训练模型（df_train为训练样本，其中y字段为标签字段）。
model.train(df_train, y_column="y")

# 使用模型进行预测（df_test为测试样本，其中y字段为标签字段）。
df_pred = model.predict(df_test, y_column="y")
print(df_pred)

# 统计模型的各项性能指标
df_eval = model.evaluate(df_test, y_column="y")
print(df_eval)
```

---

## 📁 模块说明

| 模块名               | 描述                                   |
|----------------------|--------------------------------------|
| `FADataEDA`          | 探索性数据分析                         |
| `FADataPreprocess`   | 数据预处理（标准化、异常值检测等）         |
| `FAFeatureSelect`    | 特征选择（信息图、PCA、因果性检验等）      |
| `FAModelClassify`    | 分类模型训练封装                        |
| `FAModelRegression`  | 回归模型训练封装                        |
| `FAModelCluster`     | 聚类模型训练封装                        |
| `FAModelSeries`      | 时间序列建模（自动ARIMA）               |
| `FAEvaluation`       | 模型评估与指标输出                      |
| `FAVisualize`        | 可视化模块（热图、桑基图、等高线等）      |
| `FASample`           | 样本生成与增强                        |

---

## 🧪 测试脚本示例

测试脚本位于 `tests/` 目录，支持以下演示：

- `demo_Sample.py`：样本生成与增强测试
- `demo_DataEDA.py`：数据分析演示
- `demo_DataPreprocess.py`：预处理功能测试
- `demo_FeatureSelect.py`：特征选择测试
- `demo_ModelClassify.py`：分类模型演示
- `demo_ModelRegression.py`：回归模型演示
- `demo_ModelCluster.py`：聚类模型演示
- `demo_ModelSeries.py`：时间序列建模演示
- `demo_Evaluation.py`：模型性能评估
- `demo_Visualize.py`：图形可视化测试

运行示例：

- `demo_Sample.py`：样本生成与增强测试  
  ```bash
  python tests/demo_Sample.py
  ```

- `demo_DataEDA.py`：数据分析演示  
  ```bash
  python tests/demo_DataEDA.py
  ```

- `demo_DataPreprocess.py`：预处理功能测试  
  ```bash
  python tests/demo_DataPreprocess.py
  ```

- `demo_FeatureSelect.py`：特征选择测试  
  ```bash
  python tests/demo_FeatureSelect.py
  ```

- `demo_ModelClassify.py`：分类模型演示  
  ```bash
  python tests/demo_ModelClassify.py
  ```

- `demo_ModelRegression.py`：回归模型演示  
  ```bash
  python tests/demo_ModelRegression.py
  ```

- `demo_ModelCluster.py`：聚类模型演示  
  ```bash
  python tests/demo_ModelCluster.py
  ```

- `demo_ModelSeries.py`：时间序列建模演示  
  ```bash
  python tests/demo_ModelSeries.py
  ```

- `demo_Evaluation.py`：模型性能评估  
  ```bash
  python tests/demo_Evaluation.py
  ```

- `demo_Visualize.py`：图形可视化测试  
  ```bash
  python tests/demo_Visualize.py
  ```

---

## 📄 License

FreeAeon-ML is released under the MIT License.  
© 2025 FreeAeon Contributors

---

## 🤝 欢迎贡献

欢迎 PR、Issue 与建议！请确保代码规范、清晰，附带测试。

---

## ✍️ Author

**Jim Xie**  
📧 E-Mail: [jim.xie.cn@outlook.com](mailto:jim.xie.cn@outlook.com), [xiewenwei@sina.com](mailto:xiewenwei@sina.com)  
🔗 GitHub: [https://github.com/jim-xie-cn/FreeAeon-ML](https://github.com/jim-xie-cn/FreeAeon-ML)

---

## 🧠 Citation

If you use this project in academic work, please cite it as:

> Jim Xie, *FreeAeon-ML: A comprehensive machine learning toolkit for data analysis, preprocessing, modeling, and evaluation.*, 2025.  
> GitHub Repository: https://github.com/jim-xie-cn/FreeAeon-ML
