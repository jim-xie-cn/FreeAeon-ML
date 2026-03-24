# CFADataPreprocess - 数据预处理

## 应用场景

CFADataPreprocess类提供数据预处理的核心功能,主要应用于:

- 数据正态化转换(Box-Cox变换)
- 异常值检测和移除(Z-Score方法)
- 多项式拟合和趋势分析
- 特征标准化(Z-Score/Min-Max)
- 等频分箱(分位数划分)
- Transformer位置编码

## 安装依赖

```bash
pip install FreeAeon-ML
```

或手动安装依赖包:

```bash
pip install numpy pandas scipy scikit-learn
```

## 类说明

CFADataPreprocess是一个静态方法类,提供数据预处理的各种工具方法。

## 方法详解

### 1. normal_transform - Box-Cox正态转换

将数据转换为近似正态分布(仅适用于正值数据)。

#### 参数说明

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| ds_data | pd.Series | - | 待转换数据(必须>0) |
| lambda_value | float | None | Box-Cox参数,None时自动优化 |

#### 返回值

返回元组(转换后数据, lambda值)。

#### 示例代码

```python
import pandas as pd
import numpy as np
from FreeAeonML.FADataPreprocess import CFADataPreprocess

# 生成偏态数据
np.random.seed(42)
data = pd.Series(np.random.exponential(2, 1000))

# 正态转换
transformed, lambda_val = CFADataPreprocess.normal_transform(data)
print(f"Lambda参数: {lambda_val:.4f}")
```

### 2. normal_recovery - Box-Cox逆变换

从正态分布还原到原始数据分布。

#### 参数说明

| 参数名 | 类型 | 说明 |
|--------|------|------|
| ds_data | pd.Series | 转换后的数据 |
| lambda_value | float | Box-Cox参数 |

#### 返回值

返回还原后的pd.Series。

#### 示例代码

```python
import pandas as pd
import numpy as np
from FreeAeonML.FADataPreprocess import CFADataPreprocess

data = pd.Series([1, 2, 3, 4, 5])
transformed, lambda_val = CFADataPreprocess.normal_transform(data)
recovered = CFADataPreprocess.normal_recovery(transformed, lambda_val)
print("原始数据:", data.values)
print("还原数据:", recovered.values)
```

### 3. polyfit - 多项式拟合

对时间序列或散点数据进行多项式拟合。

#### 参数说明

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| ds_y | pd.Series | - | 因变量 |
| ds_x | pd.Series | 空Series | 自变量,为空时使用索引 |
| degree | int | 2 | 多项式阶数 |

#### 返回值

返回元组(拟合对象, 残差标准差, 拟合值序列)。

#### 示例代码

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from FreeAeonML.FADataPreprocess import CFADataPreprocess

# 生成带噪声的二次函数数据
np.random.seed(42)
x = np.linspace(0, 10, 100)
y = 2*x**2 - 3*x + 1 + np.random.normal(0, 5, 100)
ds_y = pd.Series(y)

# 多项式拟合
p_obj, e_std, fitted = CFADataPreprocess.polyfit(ds_y, degree=2)

print(f"拟合多项式: {p_obj}")
print(f"残差标准差: {e_std}")

# 可视化
plt.scatter(x, y, alpha=0.5, label='原始数据')
plt.plot(x, fitted, 'r-', label='拟合曲线')
plt.legend()
plt.show()
```

### 4. get_abnormal - 获取异常值

使用Z-Score方法检测异常值。

#### 参数说明

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| ds_data | pd.Series | - | 待检测数据 |
| n_sigma | int | 3 | Z-Score阈值(标准差倍数) |

#### 返回值

返回异常值的pd.Series。

#### 示例代码

```python
import pandas as pd
import numpy as np
from FreeAeonML.FADataPreprocess import CFADataPreprocess

# 生成数据并插入异常值
np.random.seed(42)
data = pd.Series(np.random.normal(0, 1, 1000))
data.iloc[10] = 10  # 异常值
data.iloc[50] = -8  # 异常值

# 检测异常值
abnormal = CFADataPreprocess.get_abnormal(data, n_sigma=3)
print(f"检测到{len(abnormal)}个异常值:")
print(abnormal)
```

### 5. remove_abnormal - 移除异常值

移除Z-Score超过阈值的异常值。

#### 参数说明

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| ds_data | pd.Series | - | 待处理数据 |
| n_sigma | int | 3 | Z-Score阈值 |

#### 返回值

返回移除异常值后的pd.Series。

#### 示例代码

```python
import pandas as pd
import numpy as np
from FreeAeonML.FADataPreprocess import CFADataPreprocess

data = pd.Series([1, 2, 3, 100, 4, 5, -50, 6])
cleaned = CFADataPreprocess.remove_abnormal(data, n_sigma=2)
print("原始数据:", data.values)
print("清洗后:", cleaned.values)
```

### 6. get_scale - 特征标准化

对DataFrame的数值列进行标准化处理。

#### 参数说明

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| df_data | pd.DataFrame | - | 待标准化数据 |
| x_columns | list | [] | 指定列,空则自动选择数值列 |
| y_column | list | ['label'] | 标签列(不标准化) |
| scale_type | str | 'z-score' | 'z-score'或'max-min' |

#### 返回值

返回元组(标准化后DataFrame, 标准化列名列表)。

#### 示例代码

```python
import pandas as pd
import numpy as np
from FreeAeonML.FADataPreprocess import CFADataPreprocess

# 创建测试数据
df = pd.DataFrame({
    'feature1': np.random.uniform(0, 100, 100),
    'feature2': np.random.normal(50, 10, 100),
    'label': np.random.randint(0, 2, 100)
})

# Z-Score标准化
df_scaled, scaled_cols = CFADataPreprocess.get_scale(
    df, y_column=['label'], scale_type='z-score'
)
print("标准化列:", scaled_cols)
print(df_scaled.head())

# Min-Max标准化
df_minmax, _ = CFADataPreprocess.get_scale(
    df, y_column=['label'], scale_type='max-min'
)
print(df_minmax.head())
```

### 7. assign_qbins - 等频分箱

将连续变量按分位数划分为离散区间。

#### 参数说明

| 参数名 | 类型 | 说明 |
|--------|------|------|
| ds_data | pd.Series | 待分箱数据 |
| quantiles | int | 分箱数量 |

#### 返回值

返回元组(分箱标签, 区间边界列表)。

#### 示例代码

```python
import pandas as pd
import numpy as np
from FreeAeonML.FADataPreprocess import CFADataPreprocess

# 生成数据
np.random.seed(42)
data = pd.Series(np.random.normal(100, 15, 1000))

# 等频5分箱
bins, intervals = CFADataPreprocess.assign_qbins(data, quantiles=5)

print("分箱结果:")
for i, interval in enumerate(intervals):
    count = (bins == i).sum()
    print(f"箱{i}: [{interval[0]:.2f}, {interval[1]:.2f}] - {count}个样本")
```

### 8. get_transformer_position_encoding - 位置编码

生成Transformer模型的位置编码矩阵。

#### 参数说明

| 参数名 | 类型 | 说明 |
|--------|------|------|
| seq_length | int | 序列长度 |
| d_model | int | 编码维度 |

#### 返回值

返回位置编码矩阵(numpy array)。

#### 示例代码

```python
import numpy as np
import matplotlib.pyplot as plt
from FreeAeonML.FADataPreprocess import CFADataPreprocess

# 生成位置编码
pos_enc = CFADataPreprocess.get_transformer_position_encoding(
    seq_length=100, d_model=64
)

# 可视化
plt.figure(figsize=(12, 6))
plt.pcolormesh(pos_enc, cmap='RdBu')
plt.xlabel('Encoding Dimension')
plt.ylabel('Position')
plt.colorbar()
plt.title('Transformer Position Encoding')
plt.show()
```

## 完整示例

### 示例1: 数据清洗流程

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from FreeAeonML.FADataPreprocess import CFADataPreprocess

# 1. 生成带异常值和偏态的数据
np.random.seed(42)
data = pd.Series(np.random.exponential(2, 1000))
data.iloc[10] = 50  # 插入异常值
data.iloc[100] = 45

print("原始数据统计:")
print(f"均值: {data.mean():.2f}, 标准差: {data.std():.2f}")
print(f"偏度: {data.skew():.2f}")

# 2. 移除异常值
cleaned = CFADataPreprocess.remove_abnormal(data, n_sigma=3)
print(f"\n移除了{len(data) - len(cleaned)}个异常值")

# 3. Box-Cox正态转换
transformed, lambda_val = CFADataPreprocess.normal_transform(cleaned)
print(f"\nBox-Cox Lambda: {lambda_val:.4f}")
print(f"转换后偏度: {transformed.skew():.2f}")

# 4. 可视化对比
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].hist(data, bins=50, edgecolor='black')
axes[0].set_title('原始数据(含异常值)')

axes[1].hist(cleaned, bins=50, edgecolor='black')
axes[1].set_title('移除异常值后')

axes[2].hist(transformed, bins=50, edgecolor='black')
axes[2].set_title('Box-Cox转换后')

plt.tight_layout()
plt.show()
```

### 示例2: 特征工程Pipeline

```python
import pandas as pd
import numpy as np
from FreeAeonML.FADataPreprocess import CFADataPreprocess

# 生成原始数据
np.random.seed(42)
df = pd.DataFrame({
    'age': np.random.uniform(18, 80, 1000),
    'income': np.random.exponential(50000, 1000),
    'score': np.random.normal(75, 15, 1000),
    'label': np.random.randint(0, 2, 1000)
})

print("原始数据:")
print(df.describe())

# 1. 收入列正态转换
df['income_transformed'], _ = CFADataPreprocess.normal_transform(
    df['income'].clip(lower=1)
)

# 2. 年龄分箱
df['age_bin'], age_intervals = CFADataPreprocess.assign_qbins(
    df['age'], quantiles=5
)

# 3. 移除score异常值
df_cleaned = df.copy()
abnormal_idx = CFADataPreprocess.get_abnormal(df['score'], n_sigma=3).index
df_cleaned = df_cleaned.drop(abnormal_idx)

# 4. 特征标准化
df_final, scaled_cols = CFADataPreprocess.get_scale(
    df_cleaned[['age', 'income_transformed', 'score', 'label']],
    y_column=['label'],
    scale_type='z-score'
)

print("\n处理后数据:")
print(df_final.describe())
```

## 注意事项

1. **Box-Cox转换**:
   - 仅适用于严格正值数据(>0)
   - 负值或0需要先平移: `data + |min(data)| + 1`
   - lambda=0时等价于对数转换

2. **异常值检测**:
   - n_sigma=3是经验值(99.7%置信区间)
   - 小样本(<30)不建议使用Z-Score
   - 考虑使用箱线图方法(IQR)作为替代

3. **标准化选择**:
   - Z-Score: 保持分布形状,适合正态分布
   - Min-Max: 压缩到[0,1],对异常值敏感

4. **多项式拟合**:
   - degree不宜过大(推荐≤5)
   - 用于趋势分析而非预测

## 相关类链接

- [CFATransformer](./数据预处理-CFATransformer.md) - 高级数据转换类
- [CFADataTest](./数据探索-CFADataTest.md) - 数据假设检验类
- [CFACommonStats](./数据探索-CFACommonStats.md) - 统计量计算类
