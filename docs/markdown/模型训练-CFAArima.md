# CFAArima

## 功能分类
模型训练

## 类描述
ARIMA时间序列分析工具，提供模型拟合、预测和自动参数选择功能

## 应用场景

- 销售预测：预测未来一段时间的销售趋势
- 股票预测：预测股票价格走势
- 需求预测：预测产品需求变化
- 异常检测：识别时间序列中的异常模式
- 季节性分析：分解时间序列的趋势、季节性和残差
        

## 方法列表


### 主要方法

#### 1. fit(dataseries, p=1, d=0, q=1)
拟合ARIMA模型，参数(p,d,q)分别表示自回归阶数、差分阶数、移动平均阶数。

#### 2. show_model(result_arima)
显示模型摘要信息，包括AIC、BIC、参数显著性等。

#### 3. predict(result_arima)
对训练数据进行拟合预测。

#### 4. forecast(result_arima, num_future_points=1)
预测未来时间点的数值。

#### 5. show_result(train_data, predicted)
可视化显示预测结果与真实数据的对比。

#### 6. auto_fit(ds_train, ds_test, seasonal_order_range)
自动选择最优季节性ARIMA参数。

#### 7. decomposition(ds_data, model='additive', period=7)
时间序列分解，提取趋势、季节性和残差成分。
        

## 示例代码


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from FreeAeonML.FAModelSeries import CFAArima
from FreeAeonML.FADataPreprocess import CFADataPreprocess

# 1. 准备时间序列数据
np.random.seed(42)
dates = pd.date_range('2023-01-01', periods=100, freq='D')
trend = np.linspace(100, 200, 100)
seasonal = 10 * np.sin(np.arange(100) * 2 * np.pi / 7)
noise = np.random.normal(0, 5, 100)
ds_data = pd.Series(trend + seasonal + noise, index=dates)

# 移除异常值
abnormal = CFADataPreprocess.get_abnormal(ds_data, n_sigma=3)
ds_clean = ds_data.drop(abnormal.index)

# 2. 简单ARIMA拟合
model_simple = CFAArima.fit(ds_clean, p=1, d=1, q=1)
CFAArima.show_model(model_simple)

# 3. 预测
predicted = CFAArima.predict(model_simple)
CFAArima.show_result(ds_clean, predicted)

# 4. 未来预测
forecast_values = CFAArima.forecast(model_simple, num_future_points=10)
print("未来10天预测值:\n", forecast_values)

# 5. 自动参数选择
ds_train = ds_clean.head(80)
ds_test = ds_clean.tail(20)
seasonal_order_range = ((1, 3), (1, 2), (1, 3), (5, 8))
model_auto, best_order = CFAArima.auto_fit(
    ds_train,
    ds_test,
    seasonal_order_range=seasonal_order_range
)
print(f"最优参数: {best_order}")
CFAArima.show_model(model_auto)

# 6. 时间序列分解
decomposition = CFAArima.decomposition(ds_clean, model='additive', period=7)
decomposition.plot()
plt.tight_layout()
plt.show()

# 访问分解的各个成分
trend_component = decomposition.trend
seasonal_component = decomposition.seasonal
residual_component = decomposition.resid

print("\n趋势成分统计:")
print(trend_component.describe())
print("\n季节性成分统计:")
print(seasonal_component.describe())
print("\n残差成分统计:")
print(residual_component.describe())

# 7. 模型评估
predicted_test = model_auto.forecast(len(ds_test))
rmse = np.sqrt(np.mean((predicted_test - ds_test) ** 2))
print(f"\n测试集RMSE: {rmse:.4f}")
        

## 参数说明


| 方法 | 参数 | 类型 | 说明 |
|------|------|------|------|
| fit | dataseries | pd.Series | 时间序列数据 |
| | p | int | 自回归阶数 |
| | d | int | 差分阶数 |
| | q | int | 移动平均阶数 |
| forecast | result_arima | ARIMA对象 | 拟合的模型 |
| | num_future_points | int | 预测未来点数 |
| auto_fit | ds_train | pd.Series | 训练数据 |
| | ds_test | pd.Series | 测试数据 |
| | seasonal_order_range | tuple | 季节性参数范围((p),(d),(q),(period)) |
| decomposition | ds_data | pd.Series | 时间序列数据 |
| | model | str | 'additive'或'multiplicative' |
| | period | int | 季节性周期 |
        

## 返回值说明


- **fit**: ARIMA模型对象
- **predict**: 预测值序列
- **forecast**: 未来预测值
- **auto_fit**: (最优模型, 最优参数)
- **decomposition**: 分解对象（包含trend、seasonal、resid属性）
        

## 注意事项


- ARIMA适用于平稳时间序列，非平稳序列需要差分
- p表示利用过去p个时刻的值
- d表示差分次数，通常1-2次即可
- q表示利用过去q个时刻的预测误差
- AIC和BIC越小表示模型越好
- p值<0.05表示参数显著
- 数据必须为float64类型
- auto_fit会尝试所有参数组合，可能耗时较长
- 季节性分解要求period小于数据长度的一半
        

---
*生成时间: 2026-03-23 16:10:12*
*项目: FreeAeon-ML*
