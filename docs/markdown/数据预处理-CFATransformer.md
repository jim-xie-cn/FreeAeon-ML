# CFATransformer - 高级数据转换

## 应用场景

CFATransformer类提供高级的分布转换功能,主要应用于:

- 将数据从一个分布转换到另一个分布(如训练集风格转测试集风格)
- 数据增强和合成数据生成
- 域适应(Domain Adaptation)
- 分位数映射和Copula变换
- 保持或调整变量间相关性

## 安装依赖

```bash
pip install FreeAeon-ML
```

或手动安装依赖包:

```bash
pip install numpy pandas scipy
```

## 类说明

CFATransformer支持三种转换模式:
- **scale**: 线性缩放(保持分布形状)
- **quantile**: 分位数映射(改变边际分布)
- **copula**: Copula变换(改变分布并可调整相关性)

## 初始化参数

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| mode | str | 'copula' | 转换模式: 'scale'/'quantile'/'copula' |
| method | str | 'average' | 排序方法(传递给pandas.rank) |
| adjust_corr | bool | False | 是否调整相关性(仅copula模式) |

## 方法详解

### 1. fit - 学习转换映射

学习从源数据分布到目标数据分布的映射关系。

#### 参数说明

| 参数名 | 类型 | 说明 |
|--------|------|------|
| source | pd.Series/pd.DataFrame | 源数据(A分布) |
| target | pd.Series/pd.DataFrame | 目标数据(B分布) |

#### 返回值

返回self(支持链式调用)。

#### 示例代码

```python
import pandas as pd
import numpy as np
from FreeAeonML.FADataPreprocess import CFATransformer

# 准备数据
np.random.seed(42)
source = pd.Series(np.random.normal(0, 1, 100))
target = pd.Series(np.random.normal(10, 2, 200))

# 学习映射
transformer = CFATransformer(mode='quantile')
transformer.fit(source, target)
```

### 2. transform - 转换数据

将新数据从源分布转换到目标分布。

#### 参数说明

| 参数名 | 类型 | 说明 |
|--------|------|------|
| data | pd.Series/pd.DataFrame | 待转换数据(源分布风格) |

#### 返回值

返回转换后的pd.Series或pd.DataFrame。

#### 示例代码

```python
import pandas as pd
import numpy as np
from FreeAeonML.FADataPreprocess import CFATransformer

# 训练集和测试集
np.random.seed(42)
train = pd.Series(np.random.normal(0, 1, 100))
test = pd.Series(np.random.normal(0, 1, 50))
target = pd.Series(np.random.normal(10, 2, 200))

# 学习映射并转换
transformer = CFATransformer(mode='quantile')
transformer.fit(train, target)
test_transformed = transformer.transform(test)

print("原始测试集均值:", test.mean())
print("转换后均值:", test_transformed.mean())
print("目标分布均值:", target.mean())
```

### 3. inverse - 逆转换

将转换后的数据还原回源分布。

#### 参数说明

| 参数名 | 类型 | 说明 |
|--------|------|------|
| data_transformed | pd.Series/pd.DataFrame | 转换后的数据 |
| data_original | pd.Series/pd.DataFrame | 原始源数据 |

#### 返回值

返回还原后的pd.Series或pd.DataFrame。

#### 示例代码

```python
import pandas as pd
import numpy as np
from FreeAeonML.FADataPreprocess import CFATransformer

np.random.seed(42)
source = pd.Series(np.random.normal(0, 1, 100))
target = pd.Series(np.random.normal(10, 2, 100))

# 转换和逆转换
transformer = CFATransformer(mode='quantile')
transformer.fit(source, target)

transformed = transformer.transform(source)
recovered = transformer.inverse(transformed, source)

print("原始数据:", source.head().values)
print("还原数据:", recovered.head().values)
print("误差:", (source - recovered).abs().mean())
```

## 完整示例

### 示例1: 单变量分布转换对比

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from FreeAeonML.FADataPreprocess import CFATransformer

# 生成数据
np.random.seed(42)
source = pd.Series(np.random.normal(0, 1, 1000), name='source')
target = pd.Series(np.random.normal(10, 2, 1000), name='target')

# 对比三种模式
modes = ['scale', 'quantile', 'copula']
fig, axes = plt.subplots(2, 3, figsize=(15, 8))

for i, mode in enumerate(modes):
    transformer = CFATransformer(mode=mode)
    transformer.fit(source, target)
    transformed = transformer.transform(source)

    # 原始分布
    axes[0, i].hist(source, bins=50, alpha=0.5, label='Source', density=True)
    axes[0, i].hist(target, bins=50, alpha=0.5, label='Target', density=True)
    axes[0, i].set_title(f'{mode.upper()} - 原始分布')
    axes[0, i].legend()

    # 转换后分布
    axes[1, i].hist(transformed, bins=50, alpha=0.5, label='Transformed', density=True)
    axes[1, i].hist(target, bins=50, alpha=0.5, label='Target', density=True)
    axes[1, i].set_title(f'{mode.upper()} - 转换后')
    axes[1, i].legend()

plt.tight_layout()
plt.show()
```

### 示例2: 多变量Copula转换

```python
import pandas as pd
import numpy as np
from FreeAeonML.FADataPreprocess import CFATransformer

# 生成相关的多变量数据
np.random.seed(42)
n = 500

# 源数据(低相关性)
source = pd.DataFrame({
    'x': np.random.normal(0, 1, n),
    'y': np.random.normal(0, 1, n)
})

# 目标数据(高相关性)
target = pd.DataFrame({
    'x': np.random.normal(10, 2, n)
})
target['y'] = 0.8 * target['x'] + np.random.normal(0, 1, n)

print("源数据相关系数:")
print(source.corr())
print("\n目标数据相关系数:")
print(target.corr())

# Copula转换(不调整相关性)
transformer1 = CFATransformer(mode='copula', adjust_corr=False)
transformer1.fit(source, target)
trans1 = transformer1.transform(source)
print("\nCopula转换(不调整相关性):")
print(trans1.corr())

# Copula转换(调整相关性)
transformer2 = CFATransformer(mode='copula', adjust_corr=True)
transformer2.fit(source, target)
trans2 = transformer2.transform(source)
print("\nCopula转换(调整相关性):")
print(trans2.corr())
```

### 示例3: 域适应应用

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from FreeAeonML.FADataPreprocess import CFATransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 源域数据(训练集)
np.random.seed(42)
n_source = 1000
X_source = pd.DataFrame({
    'feature1': np.random.normal(0, 1, n_source),
    'feature2': np.random.normal(0, 1, n_source)
})
y_source = (X_source['feature1'] + X_source['feature2'] > 0).astype(int)

# 目标域数据(测试集,分布不同)
n_target = 500
X_target = pd.DataFrame({
    'feature1': np.random.normal(5, 2, n_target),
    'feature2': np.random.normal(5, 2, n_target)
})
y_target = (X_target['feature1'] + X_target['feature2'] > 10).astype(int)

# 方法1: 直接训练(无域适应)
model1 = LogisticRegression()
model1.fit(X_source, y_source)
y_pred1 = model1.predict(X_target)
acc1 = accuracy_score(y_target, y_pred1)

# 方法2: 域适应
transformer = CFATransformer(mode='copula', adjust_corr=True)
transformer.fit(X_target, X_source)  # 将目标域转换为源域风格
X_target_adapted = transformer.transform(X_target)

model2 = LogisticRegression()
model2.fit(X_source, y_source)
y_pred2 = model2.predict(X_target_adapted)
acc2 = accuracy_score(y_target, y_pred2)

print(f"无域适应准确率: {acc1:.4f}")
print(f"域适应后准确率: {acc2:.4f}")
```

### 示例4: 数据增强

```python
import pandas as pd
import numpy as np
from FreeAeonML.FADataPreprocess import CFATransformer

# 原始小样本数据
np.random.seed(42)
original = pd.DataFrame({
    'feature1': np.random.normal(100, 15, 50),
    'feature2': np.random.uniform(0, 100, 50)
})

# 生成参考分布
reference = pd.DataFrame({
    'feature1': np.random.normal(100, 15, 500),
    'feature2': np.random.uniform(0, 100, 500)
})

# 使用quantile模式生成更多样本
transformer = CFATransformer(mode='quantile')
transformer.fit(reference, original)

# 从参考分布采样并转换
augmented_samples = []
for _ in range(5):  # 生成5批增强数据
    sample = reference.sample(n=50)
    aug = transformer.transform(sample)
    augmented_samples.append(aug)

augmented = pd.concat(augmented_samples, ignore_index=True)

print(f"原始样本数: {len(original)}")
print(f"增强后样本数: {len(augmented)}")
print(f"\n原始数据统计:")
print(original.describe())
print(f"\n增强数据统计:")
print(augmented.describe())
```

### 示例5: 分布可视化对比

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from FreeAeonML.FADataPreprocess import CFATransformer

# 创建不同分布
np.random.seed(42)
source = pd.Series(np.random.exponential(1, 1000))
target = pd.Series(np.random.normal(5, 2, 1000))

# 三种模式转换
results = {}
for mode in ['scale', 'quantile', 'copula']:
    transformer = CFATransformer(mode=mode)
    transformer.fit(source, target)
    results[mode] = transformer.transform(source)

# 可视化
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 原始分布
axes[0, 0].hist(source, bins=50, alpha=0.7, label='Source (Exponential)', density=True)
axes[0, 0].hist(target, bins=50, alpha=0.7, label='Target (Normal)', density=True)
axes[0, 0].set_title('原始分布')
axes[0, 0].legend()

# 各模式转换结果
for i, (mode, data) in enumerate(results.items()):
    ax = axes.flat[i + 1]
    ax.hist(data, bins=50, alpha=0.7, label=f'{mode.upper()}', density=True)
    ax.hist(target, bins=50, alpha=0.7, label='Target', density=True)
    ax.set_title(f'{mode.upper()} 转换')
    ax.legend()

    # 计算KS距离
    from scipy.stats import ks_2samp
    ks_stat, p_value = ks_2samp(data, target)
    ax.text(0.02, 0.98, f'KS={ks_stat:.4f}', transform=ax.transAxes,
            verticalalignment='top')

plt.tight_layout()
plt.show()
```

## 模式对比

| 模式 | 边际分布 | 相关性 | 适用场景 |
|------|---------|-------|---------|
| scale | 不变(仅缩放) | 不变 | 简单的均值/方差调整 |
| quantile | 完全改变 | 不变 | 单变量分布转换 |
| copula | 完全改变 | 可选调整 | 多变量分布转换,域适应 |

## 注意事项

1. **数据要求**:
   - source和target必须类型一致(Series或DataFrame)
   - DataFrame时列名和数量必须相同
   - 数据长度可以不同

2. **模式选择**:
   - scale: 最快,仅改变位置和尺度
   - quantile: 适中,改变边际分布但保持相关性
   - copula: 最慢,功能最强大

3. **相关性调整**:
   - adjust_corr=True时计算量增加
   - 需要足够样本才能稳定估计相关性

4. **逆转换精度**:
   - quantile和copula模式的逆转换是近似的
   - 依赖于排序方法(method参数)

## 相关类链接

- [CFADataPreprocess](./数据预处理-CFADataPreprocess.md) - 基础数据预处理
- [CFAFeatureSelect](./特征工程-CFAFeatureSelect.md) - 特征选择类
