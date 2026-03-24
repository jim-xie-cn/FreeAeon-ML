# CFAModelClassify - 自动化分类模型训练类

## 应用场景

CFAModelClassify类提供基于H2O的自动化分类模型训练功能，适用于：

- 二分类和多分类任务
- 自动训练多种分类算法并对比
- 模型持久化保存和加载
- 批量预测和模型评估
- 特征重要性分析

## 安装依赖

```bash
pip install h2o pandas numpy scikit-learn matplotlib seaborn
```

## 类说明

### CFAModelClassify

自动化分类模型训练类，内置多种分类算法，支持模型训练、评估、保存和加载。

**支持的算法**：
- Decision Tree (DT)
- Support Vector Machine (SVM)
- Random Forest (RF)
- Artificial Neural Network (ANN)
- Naive Bayes
- Generalized Linear Model (GLM)
- Gradient Boosting Machine (GBM)
- XGBoost (如果平台支持)

## 初始化

### `__init__(ip='localhost', port=54321, models=None)`

**参数**：
- `ip` (str): H2O服务器IP地址
- `port` (int): H2O服务器端口
- `models` (dict): 自定义模型字典，默认为None使用内置模型

**示例**：
```python
from FreeAeonML.FAModelClassify import CFAModelClassify
import h2o

# 初始化H2O
h2o.init(nthreads=-1, verbose=False)

# 创建分类模型对象
model = CFAModelClassify()
```

## 方法详解

### 1. train - 训练模型

**功能**：使用训练数据训练所有内置的分类模型。

**调用参数**：
- `df_sample` (DataFrame): 训练数据集
- `x_columns` (list, 默认=[]): 特征列名列表，为空则自动推断
- `y_column` (str, 默认='label'): 目标列名
- `train_ratio` (float, 默认=0.85): 训练集比例，设为0则不划分验证集

**返回值**：无

**示例代码**：
```python
from FreeAeonML.FAModelClassify import CFAModelClassify
from FreeAeonML.FASample import CFASample
import h2o

# 初始化H2O
h2o.init(nthreads=-1, verbose=False)

# 生成训练数据
df_train = CFASample.get_random_classification(1000, n_feature=10, n_class=2)

# 创建模型并训练
model = CFAModelClassify()
model.train(df_train, y_column='y', train_ratio=0.85)
```

### 2. save - 保存模型

**功能**：将训练好的模型保存到指定目录。

**调用参数**：
- `model_path` (str): 模型保存路径

**返回值**：无

**示例代码**：
```python
# 保存模型到./models目录
model.save("./models")
```

### 3. load - 加载模型

**功能**：从指定目录加载已保存的模型。

**调用参数**：
- `model_path` (str): 模型保存路径

**返回值**：无

**示例代码**：
```python
# 加载模型
model_loaded = CFAModelClassify()
model_loaded.load("./models")
```

### 4. predict - 预测

**功能**：使用训练好的模型进行预测。

**调用参数**：
- `df_sample` (DataFrame): 待预测的数据集
- `x_columns` (list, 默认=[]): 特征列名列表
- `y_column` (str, 默认='label'): 目标列名（如果存在）

**返回值**：
- DataFrame: 包含预测结果的数据框
  - `predict`: 预测类别
  - `p0`, `p1`, ...: 各类别的概率
  - `model`: 模型名称
  - `true`: 真实标签（如果提供）

**示例代码**：
```python
# 预测
df_test = CFASample.get_random_classification(200, n_feature=10, n_class=2)
df_pred = model.predict(df_test, y_column='y')

print(df_pred.head())
print(df_pred.groupby('model')['predict'].value_counts())
```

### 5. evaluate - 评估模型

**功能**：评估所有模型的性能指标。

**调用参数**：
- `df_sample` (DataFrame): 测试数据集
- `x_columns` (list, 默认=[]): 特征列名列表
- `y_column` (str, 默认='label'): 目标列名
- `average` (str, 默认='weighted'): 多分类指标的平均方式

**返回值**：
- DataFrame: 包含各模型评估指标
  - `model`: 模型名称
  - `confusion_matrix`: 混淆矩阵
  - `recall`: 召回率
  - `mcc`: Matthews相关系数
  - `accuracy`: 准确率
  - `precision`: 精确率
  - `auc`: AUC (二分类)
  - `auc_ovo`, `auc_ovr`: AUC (多分类)
  - `f1_score`: F1分数
  - `fbeta_score`: F-beta分数

**示例代码**：
```python
# 评估模型
df_evaluate = model.evaluate(df_test, y_column='y', average='weighted')

print(df_evaluate[['model', 'accuracy', 'precision', 'recall', 'f1_score', 'auc']])

# 找出最佳模型
best_model = df_evaluate.loc[df_evaluate['accuracy'].idxmax(), 'model']
print(f"最佳模型: {best_model}")
```

### 6. importance - 特征重要性

**功能**：获取各模型的特征重要性排序。

**调用参数**：无

**返回值**：
- DataFrame: 特征重要性数据
  - `variable`: 特征名称
  - `relative_importance`: 相对重要性
  - `scaled_importance`: 标准化重要性
  - `percentage`: 百分比
  - `model`: 模型名称

**示例代码**：
```python
# 获取特征重要性
df_importance = model.importance()

print(df_importance.head(20))

# 查看特定模型的特征重要性
df_rf_importance = df_importance[df_importance['model'] == 'rf']
print(df_rf_importance.sort_values('relative_importance', ascending=False))
```

## 完整示例

### 示例1：二分类完整流程

```python
from FreeAeonML.FAModelClassify import CFAModelClassify
from FreeAeonML.FASample import CFASample
import h2o
import pandas as pd

# 1. 初始化H2O
h2o.init(nthreads=-1, verbose=False)

# 2. 生成数据
df_data = CFASample.get_random_classification(1000, n_feature=10, n_class=2)

# 3. 划分数据集
df_train, df_test = CFASample.split_dataset(df_data, test_ratio=0.2)
print(f"训练集: {df_train.shape}, 测试集: {df_test.shape}")

# 4. 训练模型
model = CFAModelClassify()
model.train(df_train, y_column='y', train_ratio=0.85)

# 5. 保存模型
model.save("./my_models")
print("模型已保存")

# 6. 加载模型（演示）
model_loaded = CFAModelClassify()
model_loaded.load("./my_models")

# 7. 预测
df_pred = model_loaded.predict(df_test, y_column='y')
print("\n预测结果示例:")
print(df_pred.head(10))

# 8. 评估
df_evaluate = model_loaded.evaluate(df_test, y_column='y')
print("\n模型评估结果:")
print(df_evaluate[['model', 'accuracy', 'precision', 'recall', 'f1_score', 'auc']])

# 9. 特征重要性
df_importance = model_loaded.importance()
print("\n特征重要性TOP 10:")
print(df_importance.head(10))

# 10. 找出最佳模型
best_model = df_evaluate.loc[df_evaluate['f1_score'].idxmax(), 'model']
best_accuracy = df_evaluate.loc[df_evaluate['f1_score'].idxmax(), 'accuracy']
print(f"\n最佳模型: {best_model} (Accuracy: {best_accuracy:.4f})")
```

### 示例2：多分类任务

```python
from FreeAeonML.FAModelClassify import CFAModelClassify
from FreeAeonML.FASample import CFASample
import h2o

# 初始化
h2o.init(nthreads=-1, verbose=False)

# 生成4分类数据
df_data = CFASample.get_random_classification(2000, n_feature=15, n_class=4)
df_train, df_test = CFASample.split_dataset(df_data, test_ratio=0.2)

print(f"类别分布:\n{df_train['y'].value_counts()}")

# 训练模型
model = CFAModelClassify()
model.train(df_train, y_column='y')

# 评估多分类性能
df_evaluate = model.evaluate(df_test, y_column='y', average='macro')
print("\n多分类评估结果:")
print(df_evaluate[['model', 'accuracy', 'precision', 'recall', 'f1_score', 'auc_ovo', 'auc_ovr']])

# 查看混淆矩阵
for idx, row in df_evaluate.iterrows():
    print(f"\n{row['model']} 混淆矩阵:")
    print(row['confusion_matrix'])
```

### 示例3：自定义模型

```python
from FreeAeonML.FAModelClassify import CFAModelClassify
from FreeAeonML.FASample import CFASample
from h2o.estimators import H2ORandomForestEstimator, H2OGradientBoostingEstimator
import h2o

# 初始化
h2o.init(nthreads=-1, verbose=False)

# 自定义模型配置
custom_models = {
    "rf_custom": H2ORandomForestEstimator(ntrees=100, max_depth=10),
    "gbm_custom": H2OGradientBoostingEstimator(ntrees=50, max_depth=5)
}

# 使用自定义模型
model = CFAModelClassify(models=custom_models)

# 训练
df_train = CFASample.get_random_classification(1000, n_feature=10, n_class=2)
model.train(df_train, y_column='y')

# 评估
df_test = CFASample.get_random_classification(200, n_feature=10, n_class=2)
df_evaluate = model.evaluate(df_test, y_column='y')
print(df_evaluate)
```

## 评估指标说明

| 指标 | 说明 | 取值范围 | 最佳值 |
|------|------|----------|--------|
| **Accuracy** | 准确率，预测正确的比例 | [0, 1] | 1 |
| **Precision** | 精确率，预测为正的样本中真正为正的比例 | [0, 1] | 1 |
| **Recall** | 召回率，真正为正的样本中被预测为正的比例 | [0, 1] | 1 |
| **F1 Score** | Precision和Recall的调和平均 | [0, 1] | 1 |
| **AUC** | ROC曲线下面积 | [0, 1] | 1 |
| **MCC** | Matthews相关系数 | [-1, 1] | 1 |

## 注意事项

1. **H2O初始化**：使用前必须初始化H2O
   ```python
   import h2o
   h2o.init(nthreads=-1, verbose=False)
   ```

2. **WSL环境**：在WSL环境下，使用`h2o.connect()`替代`h2o.init()`

3. **XGBoost可用性**：XGBoost在Windows和ARM平台可能不可用

4. **数据格式**：目标列会自动转换为因子类型

5. **内存管理**：大数据集训练时注意内存占用

6. **模型保存**：模型按算法名称分目录保存

7. **预测缺失列**：如果测试数据缺少训练时的特征，会自动填充0

## 平台兼容性

| 平台 | DT/SVM/RF/ANN/Bayes/GLM/GBM | XGBoost |
|------|---------------------------|---------|
| Linux (x86_64) | ✅ | ✅ |
| macOS (Intel) | ✅ | ✅ |
| macOS (ARM) | ✅ | ❌ |
| Windows | ✅ | ❌ |
| WSL | ✅ | ✅ (需h2o.connect) |

## 相关类

- [CFASample](数据工具-CFASample.md) - 数据生成和划分
- [CFAFeatureSelect](特征工程-CFAFeatureSelect.md) - 特征选择
- [CFAEvaluation](模型评估-CFAEvaluation.md) - 模型评估
- [CFAModelRegression](模型训练-CFAModelRegression.md) - 回归模型
