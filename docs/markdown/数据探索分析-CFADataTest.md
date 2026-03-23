# CFADataTest

## 功能分类
数据探索分析

## 类描述
数据假设检验类，提供平稳性检验、格兰杰因果检验、自相关分析等功能

## 应用场景

- 时间序列分析前需要检验数据平稳性
- 分析变量之间的因果关系
- 查看时间序列的自相关特征
        

## 方法列表


### stationarity_test(timeSeries)
ADF平稳性检验。

### show_acf_pacf(timeSeries)
显示自相关和偏自相关图。

### granger_test(ds_result, ds_source, maxlag, p_value=0.05)
格兰杰因果检验。
        

## 示例代码


from FreeAeonML.FADataEDA import CFADataTest
import numpy as np

ts_data = np.random.randn(1000)
result = CFADataTest.stationarity_test(ts_data)
CFADataTest.show_acf_pacf(ts_data)
        

## 参数说明


| 参数名 | 类型 | 说明 |
|--------|------|------|
| timeSeries | Series | 时间序列数据 |
| maxlag | int | 最大滞后期 |
        

## 返回值说明

返回检验统计量或可视化图表

## 注意事项


- 平稳序列才能使用ARIMA模型
- ADF检验p值<0.05表示序列平稳
        

---
*生成时间: 2026-03-23 16:10:12*
*项目: FreeAeon-ML*
