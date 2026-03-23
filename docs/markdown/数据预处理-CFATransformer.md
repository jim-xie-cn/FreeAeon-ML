# CFATransformer

## 功能分类
数据预处理

## 类描述
数据风格转换工具，支持将一组数据的分布特征转换为另一组数据的分布特征

## 应用场景

- 数据增强：生成与目标数据分布相似的新数据
- 域适应：将训练数据分布调整为测试数据分布
- 模拟数据生成：根据真实数据生成模拟数据
- 分布匹配：使多个数据源具有一致的统计特性
- 时间序列对齐：将历史数据分布转换为当前数据分布
        

## 方法列表


### 主要方法

#### 1. __init__(mode='copula', method='average', adjust_corr=False)
初始化转换器，选择转换模式。

#### 2. fit(source, target)
学习从源数据到目标数据的映射关系。

#### 3. transform(data)
将数据转换为目标分布风格。

#### 4. inverse(data_transformed, data_original)
将转换后的数据还原回原始分布。

### 转换模式

- **scale**: 线性缩放，仅调整均值和标准差
- **quantile**: 分位数映射，改变边际分布
- **copula**: Copula变换，同时调整边际分布和相关性结构
        

## 示例代码


import pandas as pd
import numpy as np
from FreeAeonML.FADataPreprocess import CFATransformer

# 准备源数据和目标数据
df_source = pd.DataFrame({
    'x': np.random.normal(0, 1, 1000),
    'y': np.random.exponential(1, 1000)
})
df_target = pd.DataFrame({
    'x': np.random.normal(10, 2, 1000),
    'y': np.random.exponential(5, 1000)
})

# 1. 线性缩放模式
transformer_scale = CFATransformer(mode='scale')
transformer_scale.fit(df_source, df_target)
df_scaled = transformer_scale.transform(df_source)
print("线性缩放结果:\n", df_scaled.describe())

# 2. 分位数映射模式
transformer_quantile = CFATransformer(mode='quantile')
transformer_quantile.fit(df_source, df_target)
df_quantile = transformer_quantile.transform(df_source)
print("分位数映射结果:\n", df_quantile.describe())

# 3. Copula变换模式（保留相关性）
transformer_copula = CFATransformer(mode='copula', adjust_corr=True)
transformer_copula.fit(df_source, df_target)
df_copula = transformer_copula.transform(df_source)
print("Copula变换结果:\n", df_copula.describe())

# 4. 数据还原
df_recovered = transformer_copula.inverse(df_copula, df_source)
print("还原结果:\n", df_recovered.describe())

# 5. 单变量转换
ds_source = df_source['x']
ds_target = df_target['x']
transformer_single = CFATransformer(mode='copula')
transformer_single.fit(ds_source, ds_target)
ds_transformed = transformer_single.transform(ds_source)
print("单变量转换结果:", ds_transformed.describe())
        

## 参数说明


| 方法 | 参数 | 类型 | 说明 |
|------|------|------|------|
| __init__ | mode | str | 转换模式：'scale', 'quantile', 'copula' |
| | method | str | 排序方法，默认'average' |
| | adjust_corr | bool | 是否调整相关性（仅copula模式） |
| fit | source | pd.Series/DataFrame | 源数据 |
| | target | pd.Series/DataFrame | 目标数据 |
| transform | data | pd.Series/DataFrame | 待转换数据 |
| inverse | data_transformed | pd.Series/DataFrame | 转换后的数据 |
| | data_original | pd.Series/DataFrame | 原始数据 |
        

## 返回值说明


- **fit**: 返回self，支持链式调用
- **transform**: 转换后的数据（与输入类型一致）
- **inverse**: 还原后的数据（与输入类型一致）
        

## 注意事项


- scale模式最快但仅改变位置和尺度
- quantile模式可改变分布形状但不保留相关性
- copula模式可同时调整分布和相关性但计算较慢
- adjust_corr=True时会调整变量间的相关性结构
- 源数据和目标数据的维度必须一致
- 转换后的数据统计特性接近目标数据
        

---
*生成时间: 2026-03-23 16:10:12*
*项目: FreeAeon-ML*
