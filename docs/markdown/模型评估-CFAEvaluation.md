# CFAEvaluation - 模型评估与相似度计算

## 应用场景

本模块包含两个类:
- **CFAEvaluation**: 分类模型评估和ROC曲线
- **CFASimilarity**: 序列相似度计算

主要应用于:
- 分类模型性能评估
- ROC曲线绘制和AUC计算
- 时间序列相似度比较
- 分布一致性检验

## 安装依赖

```bash
pip install FreeAeon-ML
```

## CFAEvaluation类

### 1. evaluate - 分类评估

```python
@staticmethod
def evaluate(ds_pred, ds_true, average='weighted')
```

计算分类模型的评估指标。

**参数**:
- ds_pred: 预测标签
- ds_true: 真实标签
- average: 多分类平均方式('weighted','macro','micro')

**返回指标**:
- confusion_matrix: 混淆矩阵
- accuracy: 准确率
- precision: 精确率
- recall: 召回率
- f1_score: F1分数
- auc: AUC值(二分类)
- mcc: Matthews相关系数

**示例**:
```python
import pandas as pd
from FreeAeonML.FAEvaluation import CFAEvaluation

y_true = pd.Series([0, 1, 1, 0, 1, 0])
y_pred = pd.Series([0, 1, 0, 0, 1, 1])

result = CFAEvaluation.evaluate(y_pred, y_true)
print(result)
```

### 2. get_binary_roc - 计算ROC曲线

```python
@staticmethod
def get_binary_roc(ds_true, ds_prob)
```

计算二分类ROC曲线数据。

**参数**:
- ds_true: 真实标签(0/1)
- ds_prob: 预测概率

**返回**: (auc值, ROC数据DataFrame)

### 3. show_binary_roc - 绘制ROC曲线

```python
@staticmethod
def show_binary_roc(roc_auc, df_roc, title=None)
```

**示例**:
```python
import numpy as np
import pandas as pd
from FreeAeonML.FAEvaluation import CFAEvaluation

# 模拟预测概率
y_true = pd.Series([0, 0, 1, 1, 0, 1, 1, 0, 1, 0])
y_prob = pd.Series([0.1, 0.3, 0.8, 0.7, 0.2, 0.9, 0.6, 0.4, 0.85, 0.15])

# 计算ROC
auc_score, df_roc = CFAEvaluation.get_binary_roc(y_true, y_prob)
print(f"AUC: {auc_score:.4f}")

# 绘制ROC曲线
CFAEvaluation.show_binary_roc(auc_score, df_roc, title="My Model")
```

## CFASimilarity类

计算两个序列的相似度。

### 初始化

```python
sim = CFASimilarity(list1, list2)
```

### 支持的相似度方法

| 方法 | 说明 | 取值范围 | 相似度解释 |
|------|------|---------|----------|
| Cosine() | 余弦相似度 | [0,1] | 越接近1越相似 |
| Pearson() | 皮尔森相关系数 | [-1,1] | 越接近1越正相关 |
| Euclidean() | 欧氏距离 | [0,∞) | 越小越相似 |
| Manhattan() | 曼哈顿距离 | [0,∞) | 越小越相似 |
| EMD() | 地球移动距离 | [0,∞) | 越小越相似 |
| KSTest() | KS检验 | (统计量,p值) | p值>0.05相似 |

**示例**:
```python
from FreeAeonML.FAEvaluation import CFASimilarity

list1 = [1, 2, 3, 4, 5]
list2 = [1.1, 2.2, 2.9, 4.1, 5.2]

sim = CFASimilarity(list1, list2)

print(f"余弦相似度: {sim.Cosine():.4f}")
print(f"皮尔森系数: {sim.Pearson():.4f}")
print(f"欧氏距离: {sim.Euclidean():.4f}")
print(f"曼哈顿距离: {sim.Manhattan():.4f}")
print(f"EMD距离: {sim.EMD():.4f}")

ks_stat, p_value = sim.KSTest()
print(f"KS检验: statistic={ks_stat:.4f}, p-value={p_value:.4f}")
```

## 完整示例

### 示例1: 分类模型评估

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from FreeAeonML.FAEvaluation import CFAEvaluation
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# 生成数据
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# 训练模型
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# 评估
result = CFAEvaluation.evaluate(pd.Series(y_pred), pd.Series(y_test))
print("评估结果:")
for key, value in result.iloc[0].items():
    if key != 'confusion_matrix':
        print(f"  {key}: {value:.4f}")

print("\n混淆矩阵:")
print(result.iloc[0]['confusion_matrix'])

# ROC曲线
auc_score, df_roc = CFAEvaluation.get_binary_roc(pd.Series(y_test), pd.Series(y_prob))
CFAEvaluation.show_binary_roc(auc_score, df_roc, title="Random Forest")
plt.show()
```

### 示例2: 时间序列相似度分析

```python
import numpy as np
import pandas as pd
from FreeAeonML.FAEvaluation import CFASimilarity

# 生成三个时间序列
np.random.seed(42)
ts1 = np.sin(np.linspace(0, 4*np.pi, 100)) + np.random.randn(100) * 0.1
ts2 = np.sin(np.linspace(0, 4*np.pi, 100)) + np.random.randn(100) * 0.1  # 相似
ts3 = np.cos(np.linspace(0, 4*np.pi, 100)) + np.random.randn(100) * 0.1  # 不相似

# 计算相似度
sim12 = CFASimilarity(ts1, ts2)
sim13 = CFASimilarity(ts1, ts3)

print("ts1 vs ts2 (相似序列):")
print(f"  余弦相似度: {sim12.Cosine():.4f}")
print(f"  皮尔森系数: {sim12.Pearson():.4f}")
print(f"  欧氏距离: {sim12.Euclidean():.4f}")

print("\nts1 vs ts3 (不相似序列):")
print(f"  余弦相似度: {sim13.Cosine():.4f}")
print(f"  皮尔森系数: {sim13.Pearson():.4f}")
print(f"  欧氏距离: {sim13.Euclidean():.4f}")
```

## 注意事项

1. **evaluate方法**: 要求预测值和真实值非空且类别数>1
2. **ROC曲线**: 仅适用于二分类问题
3. **相似度计算**: 两个序列长度必须相同
4. **KS检验**: p值<0.05表示分布显著不同

## 相关类链接

- [CFAModelClassify](./模型训练-CFAModelClassify.md)
- [CFAModelRegression](./模型训练-CFAModelRegression.md)
