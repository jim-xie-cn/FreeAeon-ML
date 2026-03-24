# CFACommonStats - 常见统计量计算

## 应用场景

CFACommonStats类用于计算常见的统计指标,主要应用于:

- 数据探索性分析(EDA)
- 数据质量评估(缺失率、异常值)
- 分布特征分析(偏度、峰度)
- 变量相关性分析
- 时间序列自相关分析

## 安装依赖

```bash
pip install FreeAeon-ML
```

或手动安装依赖包:

```bash
pip install numpy pandas
```

## 类说明

CFACommonStats是一个静态方法类,提供两个核心方法用于计算统计量,无需实例化即可使用。

## 方法详解

### 1. get_one_stats - 计算单个序列的详细统计量

计算单个Series或DataFrame的全面统计指标。

#### 参数说明

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| x | pd.Series/pd.DataFrame | - | 待分析的数据 |
| q | tuple | (0.25, 0.5, 0.75) | 分位数列表 |
| autocorr_lags | tuple | (1,) | 自相关滞后阶数 |
| add_corr | bool | False | 是否添加相关系数矩阵(仅DataFrame) |

#### 返回值

返回pd.DataFrame,包含以下统计量:

**基本统计量**:
- count: 非空值数量
- n: 总样本数
- n_missing: 缺失值数量
- missing_rate: 缺失率
- nunique: 唯一值数量

**数值型数据统计量**:
- mean: 均值
- median: 中位数
- mode: 众数
- min/max: 最小值/最大值
- range: 极差(max - min)
- q_0.25/q_0.5/q_0.75: 四分位数
- iqr: 四分位距(Q3 - Q1)
- var: 方差
- std: 标准差
- cv: 变异系数(std/mean)
- mad_mean: 平均绝对偏差
- skew: 偏度(Fisher定义)
- kurt_fisher: 峰度(Fisher定义)
- kurt_pearson: 峰度(Pearson定义)
- autocorr_lag1: 1阶自相关系数

**分类型数据统计量**:
- mode: 众数

#### 示例代码

```python
import pandas as pd
import numpy as np
from FreeAeonML.FADataEDA import CFACommonStats

# 单个Series分析
np.random.seed(42)
data = pd.Series(np.random.randn(1000))
stats = CFACommonStats.get_one_stats(data)
print(stats)

# DataFrame分析
df = pd.DataFrame({
    'A': np.random.randn(1000),
    'B': np.random.randint(0, 100, 1000),
    'C': np.random.choice(['X', 'Y', 'Z'], 1000)
})
stats_df = CFACommonStats.get_one_stats(df)
print(stats_df)
```

### 2. get_stats - 计算多列数据的核心统计量

快速计算DataFrame中数值型列的核心统计指标。

#### 参数说明

| 参数名 | 类型 | 说明 |
|--------|------|------|
| df_data | pd.DataFrame | 待分析的数据 |

#### 返回值

返回dict,键为列名,值为包含以下指标的字典:
- mean: 均值
- std: 标准差
- cv: 变异系数
- skew: 偏度
- kurt_fisher: 峰度(Fisher)
- kurt_pearson: 峰度(Pearson)
- autocorr_lag1: 1阶自相关
- min/max: 最小值/最大值
- range: 极差
- iqr: 四分位距

#### 示例代码

```python
import pandas as pd
import numpy as np
from FreeAeonML.FADataEDA import CFACommonStats

# 创建测试数据
df = pd.DataFrame({
    'sales': np.random.normal(100, 20, 1000),
    'price': np.random.uniform(10, 100, 1000),
    'quantity': np.random.poisson(50, 1000)
})

# 计算统计量
stats = CFACommonStats.get_stats(df)

# 打印结果
for col, metrics in stats.items():
    print(f"\n{col}列统计量:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
```

## 完整示例

### 示例1: 单列数据分析

```python
import pandas as pd
import numpy as np
from FreeAeonML.FADataEDA import CFACommonStats

# 生成正态分布数据
np.random.seed(42)
data = pd.Series(np.random.normal(loc=100, scale=15, size=1000), name='score')

# 计算统计量
stats = CFACommonStats.get_one_stats(data)

# 格式化输出
print("=" * 60)
print("数据统计分析报告")
print("=" * 60)
stats_dict = stats.iloc[0].to_dict()

print(f"\n【样本信息】")
print(f"总样本数: {int(stats_dict['n'])}")
print(f"有效样本: {int(stats_dict['count'])}")
print(f"缺失值: {int(stats_dict['n_missing'])}")
print(f"缺失率: {stats_dict['missing_rate']:.2%}")

print(f"\n【集中趋势】")
print(f"均值: {stats_dict['mean']:.2f}")
print(f"中位数: {stats_dict['median']:.2f}")
print(f"众数: {stats_dict['mode']:.2f}")

print(f"\n【离散程度】")
print(f"标准差: {stats_dict['std']:.2f}")
print(f"方差: {stats_dict['var']:.2f}")
print(f"变异系数: {stats_dict['cv']:.4f}")
print(f"极差: {stats_dict['range']:.2f}")
print(f"四分位距: {stats_dict['iqr']:.2f}")

print(f"\n【分布形态】")
print(f"偏度: {stats_dict['skew']:.4f}")
print(f"峰度(Fisher): {stats_dict['kurt_fisher']:.4f}")
print(f"峰度(Pearson): {stats_dict['kurt_pearson']:.4f}")

print(f"\n【自相关性】")
print(f"1阶自相关系数: {stats_dict['autocorr_lag1']:.4f}")
```

### 示例2: 多列对比分析

```python
import pandas as pd
import numpy as np
from FreeAeonML.FADataEDA import CFACommonStats

# 创建多列数据
np.random.seed(42)
df = pd.DataFrame({
    'Normal': np.random.normal(0, 1, 1000),
    'Uniform': np.random.uniform(-3, 3, 1000),
    'Exponential': np.random.exponential(1, 1000),
    'Poisson': np.random.poisson(5, 1000)
})

# 计算统计量
stats = CFACommonStats.get_stats(df)

# 创建对比表
comparison = pd.DataFrame(stats).T
print("\n各分布统计量对比:")
print(comparison[['mean', 'std', 'skew', 'kurt_fisher']].round(4))
```

### 示例3: 时间序列自相关分析

```python
import pandas as pd
import numpy as np
from FreeAeonML.FADataEDA import CFACommonStats

# 生成时间序列数据
np.random.seed(42)
n = 500
ar_series = pd.Series(dtype=float)
ar_series[0] = np.random.randn()

# AR(1)过程: X(t) = 0.7*X(t-1) + noise
for i in range(1, n):
    ar_series[i] = 0.7 * ar_series[i-1] + np.random.randn()

# 计算多阶自相关
stats = CFACommonStats.get_one_stats(
    ar_series,
    autocorr_lags=(1, 2, 3, 5, 10)
)

print("\n时间序列自相关分析:")
stats_dict = stats.iloc[0].to_dict()
for key in stats_dict:
    if key.startswith('autocorr'):
        print(f"{key}: {stats_dict[key]:.4f}")
```

### 示例4: 数据质量评估

```python
import pandas as pd
import numpy as np
from FreeAeonML.FADataEDA import CFACommonStats

# 创建包含缺失值和异常值的数据
np.random.seed(42)
df = pd.DataFrame({
    'A': np.random.randn(1000),
    'B': np.random.randn(1000),
    'C': np.random.randn(1000)
})

# 引入缺失值
df.loc[df.sample(frac=0.1).index, 'A'] = np.nan
df.loc[df.sample(frac=0.05).index, 'B'] = np.nan

# 引入异常值
df.loc[df.sample(n=10).index, 'C'] = 100

# 数据质量评估
stats = CFACommonStats.get_one_stats(df)

print("\n数据质量评估报告:")
print("=" * 60)
for col in df.columns:
    col_stats = stats[stats.index == col].iloc[0]
    print(f"\n列名: {col}")
    print(f"  缺失率: {col_stats['missing_rate']:.2%}")
    print(f"  唯一值: {int(col_stats['nunique'])}")
    print(f"  变异系数: {col_stats['cv']:.4f}")
    print(f"  偏度: {col_stats['skew']:.4f}")
```

### 示例5: 相关性矩阵分析

```python
import pandas as pd
import numpy as np
from FreeAeonML.FADataEDA import CFACommonStats

# 生成相关的多变量数据
np.random.seed(42)
n = 1000
df = pd.DataFrame({
    'X1': np.random.randn(n),
    'X2': np.random.randn(n),
})
df['X3'] = 0.8 * df['X1'] + 0.3 * df['X2'] + np.random.randn(n) * 0.3
df['X4'] = -0.6 * df['X1'] + np.random.randn(n) * 0.5

# 计算统计量(包含相关系数)
stats = CFACommonStats.get_one_stats(df, add_corr=True)

# 提取相关系数行
corr_row = stats[stats.index == '__corr__']
print("\n变量相关系数:")
for col in corr_row.columns:
    if col.startswith('corr__'):
        print(f"{col}: {corr_row[col].values[0]:.4f}")
```

## 注意事项

1. **数据类型**: get_one_stats自动识别数值型和分类型数据,分别计算对应统计量
2. **缺失值处理**: 统计量计算时自动跳过缺失值(NaN)
3. **自相关计算**:
   - 数据长度必须大于滞后阶数+1
   - 标准差为0时返回NaN
4. **相关系数矩阵**: add_corr=True时仅对数值型列计算,且需要至少2列
5. **变异系数**: 当均值为0时返回NaN
6. **性能考虑**: get_stats比get_one_stats更快,但返回的统计量较少

## 统计量解读

- **偏度(skew)**:
  - < 0: 左偏(负偏),长尾在左
  - = 0: 对称分布
  - \> 0: 右偏(正偏),长尾在右

- **峰度(kurtosis)**:
  - Fisher定义: 正态分布为0
  - Pearson定义: 正态分布为3
  - 值越大,分布越尖锐,尾部越重

- **变异系数(cv)**:
  - 衡量相对离散程度
  - cv < 0.15: 弱变异
  - 0.15 ≤ cv < 0.35: 中等变异
  - cv ≥ 0.35: 强变异

## 相关类链接

- [CFADataTest](./数据探索-CFADataTest.md) - 数据假设检验类
- [CFAFitter](./数据探索-CFAFitter.md) - 数据拟合类
- [CFADataPreprocess](./数据预处理-CFADataPreprocess.md) - 数据预处理类
