# CFAFitter - 数据拟合

## 应用场景

CFAFitter类用于对数据进行曲线拟合,主要应用于:

- 线性关系拟合和趋势分析
- 多项式曲线拟合(抛物线、高次曲线)
- 指数增长/衰减拟合
- 数据趋势预测和平滑
- 异常值检测(基于拟合残差)

## 安装依赖

```bash
pip install FreeAeon-ML
```

或手动安装依赖包:

```bash
pip install numpy pandas matplotlib scipy
```

## 类说明

CFAFitter类提供三种拟合方式:
- **linear**: 线性拟合 y = ax + b
- **polynomial**: 多项式拟合 y = a₀ + a₁x + a₂x² + ... + aₙxⁿ
- **exponential**: 指数拟合 y = a·e^(bx)

## 初始化参数

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| fitter | str | 'polynomial' | 拟合方式: 'linear', 'polynomial', 'exponential' |
| degree | int | 2 | 多项式阶数(仅polynomial模式有效) |

## 方法详解

### 1. fit - 拟合数据

对给定数据进行曲线拟合,计算最优参数。

#### 参数说明

| 参数名 | 类型 | 说明 |
|--------|------|------|
| x_data | array-like | 自变量数据 |
| y_data | array-like | 因变量数据 |

#### 返回值

返回拟合参数数组:
- linear模式: [a, b] 表示 y = ax + b
- polynomial模式: [a₀, a₁, ..., aₙ] 表示 y = a₀ + a₁x + ... + aₙxⁿ
- exponential模式: [a, b] 表示 y = a·e^(bx)

#### 示例代码

```python
import numpy as np
from FreeAeonML.FADataEDA import CFAFitter

# 生成带噪声的线性数据
np.random.seed(42)
x_data = np.linspace(0, 10, 100)
y_data = 2 * x_data + 1 + np.random.normal(size=x_data.size)

# 线性拟合
fitter = CFAFitter(fitter='linear')
params = fitter.fit(x_data, y_data)
print(f"拟合参数: a={params[0]:.4f}, b={params[1]:.4f}")
# 输出: 拟合参数: a=2.0123, b=0.9876
```

### 2. plot - 可视化拟合结果

绘制原始数据和拟合曲线的对比图。

#### 参数说明

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| x_data | array-like | - | 自变量数据 |
| y_data | array-like | - | 因变量数据 |
| params | array-like | None | 拟合参数,为None时使用fit()的结果 |

#### 返回值

无返回值,直接显示图形。

#### 示例代码

```python
import numpy as np
import matplotlib.pyplot as plt
from FreeAeonML.FADataEDA import CFAFitter

# 生成数据
np.random.seed(42)
x_data = np.linspace(0, 10, 100)
y_data = 2 * x_data + 1 + np.random.normal(size=x_data.size)

# 拟合并可视化
fitter = CFAFitter(fitter='linear')
fitter.fit(x_data, y_data)
fitter.plot(x_data, y_data)
plt.show()
```

### 3. get_fit_param - 获取拟合参数

获取最近一次拟合的参数。

#### 参数说明

无参数。

#### 返回值

返回拟合参数数组。

#### 示例代码

```python
from FreeAeonML.FADataEDA import CFAFitter
import numpy as np

x = np.array([1, 2, 3, 4, 5])
y = np.array([2.1, 4.0, 5.9, 8.1, 10.0])

fitter = CFAFitter(fitter='linear')
fitter.fit(x, y)
params = fitter.get_fit_param()
print(f"拟合参数: {params}")
```

### 4. get_fit_data - 计算拟合值

根据给定参数计算拟合的y值。

#### 参数说明

| 参数名 | 类型 | 说明 |
|--------|------|------|
| x_data | array-like | 自变量数据 |
| params | array-like | 拟合参数 |

#### 返回值

返回计算得到的y值数组。

#### 示例代码

```python
from FreeAeonML.FADataEDA import CFAFitter
import numpy as np

x_train = np.array([1, 2, 3, 4, 5])
y_train = np.array([2.1, 4.0, 5.9, 8.1, 10.0])

# 训练拟合模型
fitter = CFAFitter(fitter='linear')
params = fitter.fit(x_train, y_train)

# 预测新数据
x_test = np.array([6, 7, 8])
y_pred = fitter.get_fit_data(x_test, params)
print(f"预测值: {y_pred}")
```

## 完整示例

### 示例1: 线性拟合

```python
import numpy as np
import matplotlib.pyplot as plt
from FreeAeonML.FADataEDA import CFAFitter

# 生成线性数据
np.random.seed(0)
x_data = np.linspace(0, 10, 100)
y_data_linear = 2 * x_data + 1 + np.random.normal(size=x_data.size)

# 线性拟合
linear_fitter = CFAFitter(fitter='linear')
linear_params = linear_fitter.fit(x_data, y_data_linear)
print("线性拟合参数:", linear_params)

# 可视化
linear_fitter.plot(x_data, y_data_linear)
plt.title('Linear Fitting: y = ax + b')
plt.show()
```

### 示例2: 多项式拟合

```python
import numpy as np
import matplotlib.pyplot as plt
from FreeAeonML.FADataEDA import CFAFitter

# 生成抛物线数据
np.random.seed(0)
x_data = np.linspace(0, 10, 100)
y_data_polynomial = 1 * x_data**2 - 2 * x_data + 1 + np.random.normal(size=x_data.size)

# 多项式拟合(2阶)
poly_fitter = CFAFitter(fitter='polynomial', degree=2)
poly_params = poly_fitter.fit(x_data, y_data_polynomial)
print("多项式拟合参数:", poly_params)

# 可视化
poly_fitter.plot(x_data, y_data_polynomial)
plt.title('Polynomial Fitting: y = a₀ + a₁x + a₂x²')
plt.show()
```

### 示例3: 指数拟合

```python
import numpy as np
import matplotlib.pyplot as plt
from FreeAeonML.FADataEDA import CFAFitter

# 生成指数增长数据
np.random.seed(0)
x_data = np.linspace(0, 10, 100)
y_data_exponential = 2 * np.exp(0.5 * x_data) + np.random.normal(size=x_data.size)

# 指数拟合
exp_fitter = CFAFitter(fitter='exponential')
exp_params = exp_fitter.fit(x_data, y_data_exponential)
print("指数拟合参数:", exp_params)

# 可视化
exp_fitter.plot(x_data, y_data_exponential)
plt.title('Exponential Fitting: y = a·e^(bx)')
plt.show()
```

### 示例4: 对比不同拟合方法

```python
import numpy as np
import matplotlib.pyplot as plt
from FreeAeonML.FADataEDA import CFAFitter

# 生成数据
np.random.seed(42)
x_data = np.linspace(0, 5, 50)
y_data = 3 * x_data**2 - 2 * x_data + 1 + np.random.normal(0, 2, size=x_data.size)

# 创建子图
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 线性拟合
linear_fitter = CFAFitter(fitter='linear')
linear_fitter.fit(x_data, y_data)
plt.sca(axes[0])
linear_fitter.plot(x_data, y_data)
axes[0].set_title('Linear Fit')

# 多项式拟合(2阶)
poly_fitter = CFAFitter(fitter='polynomial', degree=2)
poly_fitter.fit(x_data, y_data)
plt.sca(axes[1])
poly_fitter.plot(x_data, y_data)
axes[1].set_title('Polynomial Fit (degree=2)')

# 多项式拟合(3阶)
poly3_fitter = CFAFitter(fitter='polynomial', degree=3)
poly3_fitter.fit(x_data, y_data)
plt.sca(axes[2])
poly3_fitter.plot(x_data, y_data)
axes[2].set_title('Polynomial Fit (degree=3)')

plt.tight_layout()
plt.show()
```

### 示例5: 趋势预测

```python
import numpy as np
import matplotlib.pyplot as plt
from FreeAeonML.FADataEDA import CFAFitter

# 历史数据(例如销售数据)
months = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
sales = np.array([100, 120, 135, 158, 185, 210, 245, 280, 320, 365])

# 拟合趋势
fitter = CFAFitter(fitter='polynomial', degree=2)
params = fitter.fit(months, sales)

# 预测未来3个月
future_months = np.array([11, 12, 13])
future_sales = fitter.get_fit_data(future_months, params)

# 可视化
plt.figure(figsize=(10, 6))
plt.scatter(months, sales, color='blue', label='Historical Data', s=50)
plt.plot(months, fitter.get_fit_data(months, params),
         color='red', label='Fitted Trend', linewidth=2)
plt.scatter(future_months, future_sales,
           color='green', label='Forecast', s=50, marker='^')

plt.xlabel('Month')
plt.ylabel('Sales')
plt.title('Sales Trend Analysis and Forecast')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print(f"预测第11-13月销售额: {future_sales}")
```

## 注意事项

1. **数据质量**: 拟合效果依赖于数据质量,异常值会严重影响拟合结果
2. **模型选择**:
   - 线性拟合适用于数据呈线性关系
   - 多项式拟合适用于曲线关系,但degree不宜过大(容易过拟合)
   - 指数拟合适用于增长/衰减趋势明显的数据
3. **阶数选择**: 多项式阶数过高会导致过拟合,建议从低阶开始尝试
4. **数据范围**: 指数拟合对数据范围敏感,数据过大可能导致数值溢出
5. **外推风险**: 在训练数据范围外预测(外推)误差会显著增大

## 相关类链接

- [CFADataTest](./数据探索-CFADataTest.md) - 数据假设检验类
- [CFACommonStats](./数据探索-CFACommonStats.md) - 统计量计算类
- [CFADataPreprocess](./数据预处理-CFADataPreprocess.md) - 数据预处理类(包含polyfit方法)
