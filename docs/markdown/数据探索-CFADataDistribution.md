# CFADataDistribution - 数据分布分析类

## 应用场景

CFADataDistribution类用于数据分布的可视化和统计检验，适用于：

- 检查数据是否符合正态分布
- 对比两个数据分布是否相同
- 可视化数据的分布特征（直方图、KDE、Q-Q图）
- 数据拟合和分布识别
- 数据预处理前的探索性分析

## 安装依赖

```bash
pip install numpy pandas matplotlib scipy seaborn
```

## 类说明

### CFADataDistribution

数据分布分析工具类，提供分布可视化和统计检验的静态方法。

## 方法详解

### 1. show_dist - 显示分布图

**功能**：同时显示数据的直方图(Histogram)、核密度估计图(KDE)和Q-Q图，用于直观判断数据分布特征。

**调用参数**：
- `ds` (Series): 待分析的数据序列
- `bins` (int, 默认=10): 直方图的箱数

**返回值**：无（直接显示图形）

**示例代码**：
```python
from FreeAeonML.FADataEDA import CFADataDistribution
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 生成正态分布数据
np.random.seed(42)
data_normal = pd.Series(np.random.randn(1000))

# 显示分布图
CFADataDistribution.show_dist(data_normal, bins=30)
plt.show()
```

### 2. normal_test - 正态性检验

**功能**：使用三种统计方法检验数据是否符合正态分布。

**调用参数**：
- `ds` (Series): 待检验的数据序列
- `p_value` (float, 默认=0.05): 显著性水平阈值

**返回值**：
- dict: 包含三种检验方法的结果
  - `shapiro-wilk`: Shapiro-Wilk检验结果
  - `kolmogorov-smirnov`: Kolmogorov-Smirnov检验结果
  - `skewness-kurtosis`: DAgostino-Pearson检验结果

每种检验包含：
- `is_normal`: 是否为正态分布（字符串）
- `detail`: 检验的详细统计量
- `describe`: 检验方法的说明

**示例代码**：
```python
from FreeAeonML.FADataEDA import CFADataDistribution
import numpy as np
import pandas as pd
import json

# 生成正态分布数据
np.random.seed(42)
data_normal = pd.Series(np.random.randn(1000))

# 进行正态性检验
result = CFADataDistribution.normal_test(data_normal, p_value=0.05)

# 打印结果
print(json.dumps(result, indent=4, default=str))

# 判断是否为正态分布
for test_name, test_result in result.items():
    print(f"{test_name}: {test_result['is_normal']}")
```

**输出解释**：
- `is_normal = 'True'`: 数据符合正态分布
- `is_normal = 'False'`: 数据不符合正态分布
- p-value >= 0.05: 接受原假设，数据符合正态分布
- p-value < 0.05: 拒绝原假设，数据不符合正态分布

### 3. dist_test - 分布相似性检验

**功能**：检验两个数据集的分布是否相同。

**调用参数**：
- `ds1` (Series): 第一个数据序列
- `ds2` (Series): 第二个数据序列
- `bins` (int, 默认=50): 用于卡方检验的箱数
- `p_value` (float, 默认=0.05): 显著性水平阈值

**返回值**：
- dict: 包含两种检验方法的结果
  - `f-test`: F检验（方差分析）结果
  - `chis-test`: 卡方检验结果

每种检验包含：
- `is_same`: 两个分布是否相同（字符串）
- `detail`: 检验的详细统计量
- `describe`: 检验方法的说明

**示例代码**：
```python
from FreeAeonML.FADataEDA import CFADataDistribution
import numpy as np
import pandas as pd
import json

np.random.seed(42)

# 生成两个来自相同分布的数据
data1 = pd.Series(np.random.randn(1000))
data2 = pd.Series(np.random.randn(1000))

# 检验分布是否相同
result = CFADataDistribution.dist_test(data1, data2, bins=50, p_value=0.05)

print(json.dumps(result, indent=4, default=str))

# 判断分布是否相同
for test_name, test_result in result.items():
    print(f"{test_name}: 分布相同={test_result['is_same']}")
```

### 4. normal_fit - 正态分布拟合

**功能**：对数据进行正态分布和指数分布拟合，并可视化拟合结果。

**调用参数**：
- `ds_data` (Series): 待拟合的数据序列
- `bins` (int, 默认=100): 直方图的箱数

**返回值**：无（直接显示图形）

**示例代码**：
```python
from FreeAeonML.FADataEDA import CFADataDistribution
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)

# 生成接近正态分布的数据
data = pd.Series(np.random.randn(1000))

# 进行正态拟合并显示
CFADataDistribution.normal_fit(data, bins=50)
plt.show()
```

## 完整应用示例

### 示例1：数据分布探索

```python
from FreeAeonML.FADataEDA import CFADataDistribution
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

# 生成测试数据
np.random.seed(11)
data_normal = pd.Series(np.random.randn(1000), name='Normal')
data_exponential = pd.Series(np.random.exponential(2, 1000), name='Exponential')

# 1. 可视化分布
print("=== 正态分布数据 ===")
CFADataDistribution.show_dist(data_normal, bins=20)
plt.show()

print("=== 指数分布数据 ===")
CFADataDistribution.show_dist(data_exponential, bins=20)
plt.show()

# 2. 正态性检验
result_normal = CFADataDistribution.normal_test(data_normal)
result_exp = CFADataDistribution.normal_test(data_exponential)

print("\n正态分布数据的检验结果:")
print(json.dumps(result_normal, indent=2, default=str))

print("\n指数分布数据的检验结果:")
print(json.dumps(result_exp, indent=2, default=str))

# 3. 分布拟合
CFADataDistribution.normal_fit(data_normal, bins=50)
plt.title('Normal Distribution Fit')
plt.show()
```

### 示例2：两组数据对比

```python
from FreeAeonML.FADataEDA import CFADataDistribution
import numpy as np
import pandas as pd
import json

np.random.seed(42)

# 生成两组数据：一组来自N(0,1)，另一组来自N(0.5,1.2)
group_a = pd.Series(np.random.randn(500))
group_b = pd.Series(np.random.normal(0.5, 1.2, 500))

# 检验两组数据分布是否相同
result = CFADataDistribution.dist_test(group_a, group_b, bins=50, p_value=0.05)

print("两组数据分布对比:")
print(json.dumps(result, indent=2, default=str))

# 解读结果
if result['f-test']['is_same'] == 'True':
    print("\nF检验: 两组数据分布相同")
else:
    print("\nF检验: 两组数据分布不同")

if result['chis-test']['is_same'] == 'True':
    print("卡方检验: 两组数据分布相同")
else:
    print("卡方检验: 两组数据分布不同")
```

## 三种正态性检验方法对比

| 检验方法 | 适用场景 | 优点 | 缺点 |
|---------|---------|------|------|
| **Shapiro-Wilk** | 小样本(3≤n≤50) | 对小样本敏感 | 受异常值影响大 |
| **Kolmogorov-Smirnov** | 大样本 | 适用于任何连续分布 | 对样本量要求高 |
| **DAgostino-Pearson** | 中大样本 | 基于偏度和峰度 | 需要较大样本量 |

## 注意事项

1. **样本量**：不同检验方法对样本量有不同要求，建议样本量大于30
2. **p值解释**：p值越大，越有可能是正态分布
3. **异常值**：异常值会显著影响正态性检验结果，建议先进行异常值处理
4. **多重检验**：建议综合多种检验方法的结果进行判断
5. **可视化优先**：始终先通过可视化直观判断分布特征

## 相关类

- [CFADataTest](数据探索-CFADataTest.md) - 假设检验
- [CFAFitter](数据探索-CFAFitter.md) - 数据拟合
- [CFADataPreprocess](数据预处理-CFADataPreprocess.md) - 数据预处理
