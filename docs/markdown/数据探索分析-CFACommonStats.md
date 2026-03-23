# CFACommonStats

## 功能分类
数据探索分析

## 类描述
统计量计算类，提供全面的描述性统计分析功能

## 应用场景

- 需要计算数据的描述性统计量
- 数据质量评估和数据探索
- 特征工程前的数据理解
        

## 方法列表


### get_one_stats(x, q=(0.25, 0.5, 0.75), autocorr_lags=(1,), add_corr=False)
计算单个变量的统计量。

### get_stats(df_data)
计算DataFrame所有列的关键统计量。
        

## 示例代码


from FreeAeonML.FADataEDA import CFACommonStats
import pandas as pd
import numpy as np

df = pd.DataFrame({'col1': np.random.randn(1000)})
stats = CFACommonStats.get_stats(df)
print(stats)
        

## 参数说明


| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| x | Series/DataFrame | - | 待分析数据 |
| q | tuple | (0.25, 0.5, 0.75) | 分位数 |
        

## 返回值说明

返回包含统计量的DataFrame或字典

## 注意事项


- cv (变异系数) = std / mean
- kurt_fisher: Fisher峰度（正态分布为0）
        

---
*生成时间: 2026-03-23 16:10:12*
*项目: FreeAeon-ML*
