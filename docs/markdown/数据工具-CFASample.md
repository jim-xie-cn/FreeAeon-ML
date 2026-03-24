# CFASample - 样本生成与数据处理类

## 应用场景

CFASample类提供了机器学习中常用的样本数据生成和数据集处理功能，适用于：

- 快速生成测试数据进行算法验证
- 生成回归、分类、聚类等不同任务的随机样本数据
- 数据集的训练测试划分
- 不平衡数据的重采样处理
- 时序数据的变化点检测
- 批量数据处理

## 安装依赖

```bash
pip install pandas numpy scikit-learn imbalanced-learn
```

## 类说明

### CFASample

样本生成与数据处理工具类，提供多种数据生成和处理的静态方法。

## 方法详解

### 1. get_random_regression - 生成回归样本

**功能**：生成随机回归数据集，包含2个特征和1个目标变量。

**调用参数**：
- `n_sample` (int, 默认=100): 样本数量

**返回值**：
- DataFrame: 包含x1, x2特征列和y目标列的数据集

**示例代码**：
```python
from FreeAeonML.FASample import CFASample

# 生成100个回归样本
df_regression = CFASample.get_random_regression(n_sample=100)
print(df_regression.head())
```

### 2. get_random_classification - 生成分类样本

**功能**：生成随机分类数据集，支持多分类任务。

**调用参数**：
- `n_sample` (int, 默认=100): 样本数量
- `n_feature` (int, 默认=2): 特征数量
- `n_class` (int, 默认=2): 类别数量

**返回值**：
- DataFrame: 包含x0, x1, ..., xn特征列和y目标列的数据集

**示例代码**：
```python
from FreeAeonML.FASample import CFASample

# 生成1000个样本，10个特征，2分类
df_classification = CFASample.get_random_classification(
    n_sample=1000,
    n_feature=10,
    n_class=2
)
print(df_classification.shape)
print(df_classification['y'].value_counts())
```

### 3. get_random_cluster - 生成聚类样本

**功能**：生成随机聚类数据集，包含2个簇。

**调用参数**：
- `n_sample` (int, 默认=100): 样本数量

**返回值**：
- DataFrame: 包含x1, x2特征列和y簇标签列的数据集

**示例代码**：
```python
from FreeAeonML.FASample import CFASample

# 生成100个聚类样本
df_cluster = CFASample.get_random_cluster(n_sample=100)
print(df_cluster.head())
```

### 4. split_dataset - 划分数据集

**功能**：将数据集划分为训练集和测试集。

**调用参数**：
- `df` (DataFrame): 待划分的数据集
- `test_ratio` (float, 默认=0.2): 测试集比例

**返回值**：
- tuple: (df_train, df_test) 训练集和测试集

**示例代码**：
```python
from FreeAeonML.FASample import CFASample

# 生成数据并划分
df_sample = CFASample.get_random_classification(1000, n_feature=10, n_class=2)
df_train, df_test = CFASample.split_dataset(df_sample, test_ratio=0.2)

print(f"训练集大小: {df_train.shape}")
print(f"测试集大小: {df_test.shape}")
```

### 5. resample_smote - SMOTE重采样

**功能**：使用SMOTE算法对不平衡数据进行过采样，支持分类特征。

**调用参数**：
- `df_sample` (DataFrame): 待重采样的数据集
- `x_columns` (list, 默认=[]): 特征列名列表，为空则自动推断
- `y_column` (str, 默认='label'): 目标列名
- `random_state` (int, 默认=42): 随机种子

**返回值**：
- DataFrame: 重采样后的平衡数据集

**示例代码**：
```python
from FreeAeonML.FASample import CFASample

# 生成不平衡数据
df_imbalanced = CFASample.get_random_classification(1000, n_feature=10, n_class=2)

# SMOTE重采样
df_balanced = CFASample.resample_smote(df_imbalanced, y_column='y')

print("原始数据分布:")
print(df_imbalanced['y'].value_counts())
print("\n重采样后数据分布:")
print(df_balanced['y'].value_counts())
```

### 6. resample_balance - 简单平衡重采样

**功能**：通过上采样少数类来平衡数据集。

**调用参数**：
- `df_data` (DataFrame): 待平衡的数据集
- `y_column` (str, 默认='labels'): 目标列名

**返回值**：
- DataFrame: 平衡后的数据集

**示例代码**：
```python
from FreeAeonML.FASample import CFASample

df_sample = CFASample.get_random_classification(1000, n_feature=10, n_class=2)
df_balanced = CFASample.resample_balance(df_sample, y_column='y')

print(df_balanced['y'].value_counts())
```

### 7. find_changed_index - 查找变化索引

**功能**：在时序数据中找到值发生变化的索引位置。

**调用参数**：
- `ds` (Series): 时序数据序列

**返回值**：
- ndarray: 变化点的索引数组

**示例代码**：
```python
from FreeAeonML.FASample import CFASample
import pandas as pd

# 创建时序数据
ds = pd.Series([1, 1, 1, 2, 2, 3, 3, 3, 1])

# 查找变化点
changed_indices = CFASample.find_changed_index(ds)
print(f"变化发生在索引: {changed_indices}")
```

### 8. split_dataframe - 批量划分DataFrame

**功能**：将DataFrame按批次大小划分为多个小的DataFrame。

**调用参数**：
- `df_sample` (DataFrame): 待划分的数据集
- `batch_size` (int, 默认=10): 每批的大小

**返回值**：
- list: DataFrame列表，每个元素是一个批次

**示例代码**：
```python
from FreeAeonML.FASample import CFASample

df_sample = CFASample.get_random_classification(100, n_feature=5, n_class=2)

# 划分为批次，每批10条
batches = CFASample.split_dataframe(df_sample, batch_size=10)

print(f"总批次数: {len(batches)}")
print(f"第一批大小: {len(batches[0])}")
```

## 完整示例

```python
from FreeAeonML.FASample import CFASample

# 1. 生成不同类型的样本数据
df_classification = CFASample.get_random_classification(1000, n_feature=10, n_class=4)
df_regression = CFASample.get_random_regression(500)
df_cluster = CFASample.get_random_cluster(300)

print("分类数据:", df_classification.shape)
print("回归数据:", df_regression.shape)
print("聚类数据:", df_cluster.shape)

# 2. 数据集划分
df_train, df_test = CFASample.split_dataset(df_regression, test_ratio=0.2)
print(f"\n训练集: {df_train.shape}, 测试集: {df_test.shape}")

# 3. 处理不平衡数据
df_balanced = CFASample.resample_smote(df_classification, y_column='y')
print("\n重采样前分布:", df_classification['y'].value_counts().to_dict())
print("重采样后分布:", df_balanced['y'].value_counts().to_dict())

# 4. 批量处理
batches = CFASample.split_dataframe(df_train, batch_size=50)
print(f"\n数据被划分为 {len(batches)} 个批次")
```

## 注意事项

1. **SMOTE重采样**：适用于不平衡分类问题，但可能引入噪声，建议结合交叉验证使用
2. **随机种子**：为保证实验可重复性，建议设置random_state参数
3. **数据量**：生成大量样本时注意内存占用
4. **特征数量**：分类任务中n_informative默认等于n_feature，确保特征有效性

## 相关类

- [CFADataPreprocess](数据预处理-CFADataPreprocess.md) - 数据预处理
- [CFAModelClassify](模型训练-CFAModelClassify.md) - 分类模型训练
- [CFAModelRegression](模型训练-CFAModelRegression.md) - 回归模型训练
