# CFADataTest - 数据假设检验

## 应用场景

CFADataTest类用于时间序列数据的统计检验,主要应用于:

- 时间序列平稳性检验(ARIMA建模前的准备)
- 自相关性和偏自相关性分析
- 格兰杰因果关系检验(变量间因果关系分析)
- 白噪声序列判断

## 安装依赖

```bash
pip install FreeAeon-ML
```

或手动安装依赖包:

```bash
pip install numpy pandas matplotlib scipy statsmodels
```

## 类说明

CFADataTest是一个静态方法类,提供三个核心检验方法,无需实例化即可使用。

## 方法详解

### 1. stationarity_test - 平稳性检验

检验时间序列是否为平稳序列,用于判断是否可以使用ARIMA模型。

#### 参数说明

| 参数名 | 类型 | 说明 |
|--------|------|------|
| timeSeries | pd.Series | 时间序列数据 |

#### 返回值

返回pd.Series,包含以下指标:
- Test Statistic: ADF检验统计量
- p-value: 显著性水平(越接近0越平稳)
- Critical Value (1%/5%/10%): 临界值

#### 判断标准

满足以下条件时,序列为平稳序列:
1. Test Statistic < Critical Value (1%, 5%, 10%)
2. p-value ≈ 0 (接近0)

#### 示例代码

```python
import pandas as pd
import numpy as np
from FreeAeonML.FADataEDA import CFADataTest

# 生成测试数据
np.random.seed(42)
data = np.random.randn(100)
ts = pd.Series(data)

# 平稳性检验
result = CFADataTest.stationarity_test(ts)
print(result)

# 输出示例:
# Test Statistic              -9.876543
# p-value                      0.000000
# Critical Value (1%)         -3.496960
# Critical Value (5%)         -2.890611
# Critical Value (10%)        -2.582128
```

### 2. show_acf_pacf - 自相关和偏自相关图

绘制自相关函数(ACF)和偏自相关函数(PACF)图,用于分析时间序列的相关性特征。

#### 参数说明

| 参数名 | 类型 | 说明 |
|--------|------|------|
| timeSeries | pd.Series | 时间序列数据 |

#### 返回值

无返回值,直接显示ACF和PACF图形。

#### 图形解读

**ACF图(自相关系数)**:
- 横轴: 滞后阶数(lag)
- 纵轴: 自相关系数
- 显著性: 超过置信区间表示该滞后阶数有显著相关性
- 截尾特性: 系数逐渐趋于0表示序列平稳,可用ARMA模型

**PACF图(偏自相关系数)**:
- 横轴: 滞后阶数(lag)
- 纵轴: 偏自相关系数
- 截尾特性: 在某个滞后阶数截尾,后续快速衰减,可用AR模型
- 模型阶数: 截尾位置即为AR模型的阶数

#### 示例代码

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from FreeAeonML.FADataEDA import CFADataTest

# 生成时间序列数据
np.random.seed(42)
data = np.cumsum(np.random.randn(100))  # 随机游走
ts = pd.Series(data)

# 显示ACF和PACF图
CFADataTest.show_acf_pacf(ts)
plt.show()
```

### 3. granger_test - 格兰杰因果检验

检验一个时间序列是否对另一个时间序列有预测能力(因果关系)。

#### 参数说明

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| ds_result | pd.Series | - | 结果序列(因变量,必须为平稳序列) |
| ds_source | pd.Series | - | 原因序列(自变量,必须为平稳序列) |
| maxlag | int/list | - | 最大滞后阶数,整数时遍历所有lag |
| p_value | float | 0.05 | 显著性水平 |

#### 返回值

返回一个元组,包含4个元素:
1. min_lag_p (float): 最小的p值
2. best_lag (int): 最佳滞后阶数
3. approve_list (list): 通过检验的lag列表
4. detail (dict): 详细检验结果

#### 示例代码

```python
import pandas as pd
import numpy as np
from FreeAeonML.FADataEDA import CFADataTest

# 生成两个相关的时间序列
np.random.seed(42)
n = 100
source = np.random.randn(n)
result = np.zeros(n)
for i in range(2, n):
    result[i] = 0.5 * source[i-1] + 0.3 * source[i-2] + np.random.randn()

ds_source = pd.Series(source)
ds_result = pd.Series(result)

# 格兰杰因果检验
min_p, best_lag, approve_list, detail = CFADataTest.granger_test(
    ds_result=ds_result,
    ds_source=ds_source,
    maxlag=5,
    p_value=0.05
)

print(f"最小p值: {min_p}")
print(f"最佳滞后阶数: {best_lag}")
print(f"通过检验的lag: {approve_list}")
```

## 完整示例

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from FreeAeonML.FADataEDA import CFADataTest

# 1. 生成测试数据
np.random.seed(42)
n = 200

# 平稳序列
stationary_series = pd.Series(np.random.randn(n))

# 非平稳序列(随机游走)
non_stationary_series = pd.Series(np.cumsum(np.random.randn(n)))

# 2. 平稳性检验
print("=" * 50)
print("平稳序列检验结果:")
result1 = CFADataTest.stationarity_test(stationary_series)
print(result1)

print("\n" + "=" * 50)
print("非平稳序列检验结果:")
result2 = CFADataTest.stationarity_test(non_stationary_series)
print(result2)

# 3. 自相关和偏自相关分析
print("\n" + "=" * 50)
print("绘制ACF和PACF图:")
CFADataTest.show_acf_pacf(stationary_series)

# 4. 格兰杰因果检验
# 构造因果关系: result[t] = 0.7*source[t-1] + noise
source = np.random.randn(n)
result_ts = np.zeros(n)
for i in range(1, n):
    result_ts[i] = 0.7 * source[i-1] + 0.1 * np.random.randn()

ds_source = pd.Series(source)
ds_result = pd.Series(result_ts)

print("\n" + "=" * 50)
print("格兰杰因果检验:")
min_p, best_lag, approve_list, detail = CFADataTest.granger_test(
    ds_result=ds_result,
    ds_source=ds_source,
    maxlag=10,
    p_value=0.05
)

print(f"最小p值: {min_p:.6f}")
print(f"最佳滞后阶数: {best_lag}")
print(f"通过检验的lag列表:")
for item in approve_list:
    print(f"  lag={item['lag']}, max p-value={item['max p-value']:.6f}")

plt.show()
```

## 注意事项

1. **平稳性要求**: granger_test方法要求输入的序列必须是平稳序列,使用前应先进行平稳性检验
2. **数据长度**: 时间序列数据应有足够的观测值(建议至少50个以上)
3. **滞后阶数选择**: maxlag不宜过大,通常设置为数据长度的1/10左右
4. **显著性水平**: p_value默认0.05,可根据实际需求调整(0.01更严格,0.1更宽松)
5. **因果关系解释**: 格兰杰因果关系是统计意义上的预测关系,不一定代表真实的因果关系

## 相关类链接

- [CFADataPreprocess](./数据预处理-CFADataPreprocess.md) - 数据预处理类,包含平稳化处理
- [CFAFitter](./数据探索-CFAFitter.md) - 数据拟合类
- [CFACommonStats](./数据探索-CFACommonStats.md) - 统计量计算类
- [CFAArima](./模型训练-CFAArima.md) - ARIMA时间序列模型
