# CFADataDistribution

## 功能分类
数据探索分析

## 类描述
数据分布分析类，提供分布可视化、正态性检验、分布比较等功能

## 应用场景

- 需要检验数据是否服从正态分布
- 比较两组数据的分布是否相同
- 可视化数据的分布特征
- 对数据进行分布拟合
        

## 方法列表


### show_dist(ds, bins=10)
显示数据的分布图。

### normal_test(ds, p_value=0.05)
检验数据是否服从正态分布。

### dist_test(ds1, ds2, bins=50, p_value=0.05)
检验两个分布是否相同。

### normal_fit(ds_data, bins=100)
对数据进行分布拟合。
        

## 示例代码


from FreeAeonML.FADataEDA import CFADataDistribution
import numpy as np

data = np.random.randn(1000)
CFADataDistribution.show_dist(data, bins=20)
result = CFADataDistribution.normal_test(data)
print(result)
        

## 参数说明


| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| ds | Series | - | 待检验数据 |
| bins | int | 10 | 直方图分箱数 |
| p_value | float | 0.05 | 显著性水平 |
        

## 返回值说明

返回检验结果字典或可视化图表

## 注意事项


- Shapiro-Wilk检验适用于小样本
- p值大于显著性水平表示接受原假设
        

---
*生成时间: 2026-03-23 16:10:12*
*项目: FreeAeon-ML*
