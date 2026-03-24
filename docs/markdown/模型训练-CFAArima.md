# CFAArima - ARIMA时间序列模型

## 应用场景

CFAArima类提供ARIMA时间序列建模功能,主要应用于:

- 销售预测
- 股票价格预测
- 天气预测
- 能源需求预测
- 时间序列分解

## 安装依赖

```bash
pip install FreeAeon-ML
```

## 类说明

CFAArima是静态方法类,提供ARIMA建模的完整流程。

## 方法详解

### 1. fit - 拟合ARIMA模型

```python
@staticmethod
def fit(dataseries, p=1, d=0, q=1)
```

拟合ARIMA(p,d,q)模型。

**参数**:
- dataseries: 时间序列数据
- p: AR阶数
- d: 差分阶数
- q: MA阶数

**示例**:
```python
import pandas as pd
import numpy as np
from FreeAeonML.FAModelSeries import CFAArima

# 生成时间序列
data = pd.Series(np.random.randn(100).cumsum())

# 拟合ARIMA
model = CFAArima.fit(data, p=1, d=1, q=1)
print(model.summary())
```

### 2. predict - 样本内预测

```python
@staticmethod
def predict(result_arima)
```

对训练数据进行预测。

### 3. forecast - 未来预测

```python
@staticmethod
def forecast(result_arima, num_future_points=1)
```

预测未来N个时间点的值。

**示例**:
```python
# 拟合模型
model = CFAArima.fit(data, p=1, d=1, q=1)

# 样本内预测
pred = CFAArima.predict(model)

# 预测未来10个点
future = CFAArima.forecast(model, num_future_points=10)
print("未来10期预测:", future)
```

### 4. auto_fit - 自动参数优化

```python
@staticmethod
def auto_fit(ds_train, ds_test, seasonal_order_range=((1,2),(1,2),(1,10),(1,7)))
```

自动搜索最优ARIMA参数。

**参数**:
- ds_train: 训练集
- ds_test: 验证集
- seasonal_order_range: 参数搜索范围(p范围, d范围, q范围, 周期范围)

**返回**: (最优模型, 最优参数)

**示例**:
```python
# 划分训练测试集
train = data[:80]
test = data[80:]

# 自动优化参数
best_model, best_params = CFAArima.auto_fit(
    ds_train=train,
    ds_test=test,
    seasonal_order_range=((1,3),(1,2),(1,3),(5,8))
)

print(f"最优参数: {best_params}")
```

### 5. decomposition - 时间序列分解

```python
@staticmethod
def decomposition(ds_data, model='additive', period=7)
```

将时间序列分解为趋势、季节性和残差。

**参数**:
- ds_data: 时间序列
- model: 'additive'(加法模型)或'multiplicative'(乘法模型)
- period: 周期

**示例**:
```python
import matplotlib.pyplot as plt

# 分解时间序列
decomp = CFAArima.decomposition(data, model='additive', period=7)

# 可视化
decomp.plot()
plt.show()

# 提取各成分
trend = decomp.trend
seasonal = decomp.seasonal
residual = decomp.resid
```

## 完整示例

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from FreeAeonML.FAModelSeries import CFAArima

# 1. 生成带趋势和季节性的数据
np.random.seed(42)
n = 200
t = np.arange(n)
trend = 0.05 * t
seasonal = 10 * np.sin(2 * np.pi * t / 12)
noise = np.random.randn(n)
data = pd.Series(trend + seasonal + noise)

# 2. 分解时间序列
decomp = CFAArima.decomposition(data, model='additive', period=12)
decomp.plot()
plt.show()

# 3. 划分训练测试集
train = data[:160]
test = data[160:]

# 4. 自动寻找最优参数
print("自动优化ARIMA参数...")
best_model, best_params = CFAArima.auto_fit(
    ds_train=train,
    ds_test=test,
    seasonal_order_range=((1,3),(0,2),(1,3),(10,14))
)

print(f"最优参数: {best_params}")

# 5. 样本内预测
pred = CFAArima.predict(best_model)

# 6. 未来预测
future = CFAArima.forecast(best_model, num_future_points=20)

# 7. 可视化
plt.figure(figsize=(12, 6))
plt.plot(data.index, data.values, label='Original', alpha=0.7)
plt.plot(pred.index, pred.values, label='Fitted', alpha=0.7)
future_index = range(len(data), len(data) + len(future))
plt.plot(future_index, future, label='Forecast', color='red')
plt.axvline(x=len(data), color='gray', linestyle='--')
plt.legend()
plt.title('ARIMA Time Series Forecast')
plt.show()

# 8. 显示模型摘要
CFAArima.show_model(best_model)
```

## 参数选择建议

**p(AR阶数)**:
- 查看PACF图,截尾位置
- 一般p∈[0,5]

**d(差分阶数)**:
- 通过ADF检验确定
- 一阶差分通常足够
- d>2很少使用

**q(MA阶数)**:
- 查看ACF图,截尾位置
- 一般q∈[0,5]

**周期(period)**:
- 日数据: 7(周), 365(年)
- 月数据: 12(年)
- 小时数据: 24(天)

## 注意事项

1. 数据要求: 需要平稳序列或可差分为平稳
2. 参数优化: auto_fit计算量大,范围不宜过大
3. 预测精度: 短期预测较准,长期预测误差增大
4. 模型诊断: 查看残差是否为白噪声

## 相关类链接

- [CFADataTest](./数据探索-CFADataTest.md) - 平稳性检验
- [CFADataPreprocess](./数据预处理-CFADataPreprocess.md) - 数据预处理
