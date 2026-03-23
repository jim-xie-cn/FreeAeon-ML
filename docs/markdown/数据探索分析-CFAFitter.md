# CFAFitter

## 功能分类
数据探索分析

## 类描述
数据拟合类，支持线性拟合、多项式拟合和指数拟合

## 应用场景

- 需要对数据进行趋势拟合
- 预测数据未来走势
- 平滑噪声数据
        

## 方法列表


### __init__(fitter='polynomial', degree=2)
初始化拟合器。

### fit(x_data, y_data)
对数据进行拟合。

### plot(x_data, y_data, params=None)
绘制拟合曲线。
        

## 示例代码


from FreeAeonML.FADataEDA import CFAFitter
import numpy as np

x = np.linspace(0, 10, 100)
y = 2 * x + 1 + np.random.normal(size=100)

fitter = CFAFitter(fitter='linear')
params = fitter.fit(x, y)
fitter.plot(x, y)
        

## 参数说明


| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| fitter | str | 'polynomial' | 拟合类型 |
| degree | int | 2 | 多项式阶数 |
        

## 返回值说明

返回拟合参数数组

## 注意事项


- 多项式拟合避免阶数过高
- 指数拟合适合增长率恒定的数据
        

---
*生成时间: 2026-03-23 16:10:12*
*项目: FreeAeon-ML*
