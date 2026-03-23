# CFADataPreprocess

## 功能分类
数据预处理

## 类描述
数据预处理工具类，提供数据清洗、标准化、异常检测等预处理功能

## 应用场景

- 数据清洗：检测和移除数据中的异常值
- 数据转换：将数据转换为正态分布以满足模型假设
- 特征标准化：对数值特征进行Z-Score或Min-Max标准化
- 多项式拟合：对时间序列数据进行趋势拟合
- 位置编码：为Transformer模型生成位置编码
- 数据分箱：将连续数据离散化为分类数据
        

## 方法列表


### 主要方法

#### 1. normal_transform(ds_data, lambda_value=None)
使用Box-Cox变换将数据转换为正态分布。

#### 2. normal_recovery(ds_data, lambda_value)
从Box-Cox变换后的数据还原为原始分布。

#### 3. polyfit(ds_y, ds_x=pd.Series([]), degree=2)
对数据进行多项式拟合。

#### 4. get_abnormal(ds_data, n_sigma=3)
基于Z-Score检测异常数据（|z|>n_sigma）。

#### 5. remove_abnormal(ds_data, n_sigma=3)
移除异常数据，返回正常数据。

#### 6. get_scale(df_data, x_columns=[], y_column=['label'], scale_type='z-score')
对特征进行标准化处理，支持z-score和max-min两种方法。

#### 7. get_transformer_position_encoding(seq_length, d_model)
生成Transformer模型所需的位置编码。

#### 8. assign_qbins(ds_data, quantiles)
等频率分箱，将连续数据离散化。
        

## 示例代码


import pandas as pd
import numpy as np
from FreeAeonML.FADataPreprocess import CFADataPreprocess

# 1. 异常值检测和移除
ds_data = pd.Series([1, 2, 3, 4, 5, 100])  # 100是异常值
abnormal = CFADataPreprocess.get_abnormal(ds_data, n_sigma=3)
print("异常值:", abnormal)
clean_data = CFADataPreprocess.remove_abnormal(ds_data, n_sigma=3)
print("清洗后数据:", clean_data)

# 2. Box-Cox正态转换
ds_positive = pd.Series([1, 2, 3, 4, 5, 6])
transformed, lambda_val = CFADataPreprocess.normal_transform(ds_positive)
print("转换后数据:", transformed)
recovered = CFADataPreprocess.normal_recovery(transformed, lambda_val)
print("还原后数据:", recovered)

# 3. 数据标准化
df_sample = pd.DataFrame({
    'x1': [1, 2, 3, 4, 5],
    'x2': [10, 20, 30, 40, 50],
    'y': [0, 1, 0, 1, 0]
})
df_scaled, scale_cols = CFADataPreprocess.get_scale(
    df_sample, y_column=['y'], scale_type='z-score'
)
print("标准化后数据:\n", df_scaled)

# 4. 多项式拟合
ds_y = pd.Series([1, 4, 9, 16, 25])
p_obj, e_std, fitted = CFADataPreprocess.polyfit(ds_y, degree=2)
print("拟合结果:", fitted)

# 5. 位置编码
pos_enc = CFADataPreprocess.get_transformer_position_encoding(seq_length=10, d_model=64)
print("位置编码形状:", pos_enc.shape)

# 6. 等频分箱
ds_data = pd.Series(np.random.randn(1000))
bins, intervals = CFADataPreprocess.assign_qbins(ds_data, quantiles=5)
print("分箱结果:", bins.value_counts())
print("分箱区间:", intervals)
        

## 参数说明


| 方法 | 参数 | 类型 | 说明 |
|------|------|------|------|
| normal_transform | ds_data | pd.Series | 待转换数据（必须>0） |
| | lambda_value | float/None | Box-Cox参数，None时自动计算 |
| normal_recovery | ds_data | pd.Series | 转换后的数据 |
| | lambda_value | float | Box-Cox参数 |
| polyfit | ds_y | pd.Series | 因变量数据 |
| | ds_x | pd.Series | 自变量数据，默认为时间序列 |
| | degree | int | 多项式阶数 |
| get_abnormal | ds_data | pd.Series | 待检测数据 |
| | n_sigma | int | Z-Score阈值，默认3 |
| remove_abnormal | ds_data | pd.Series | 待清洗数据 |
| | n_sigma | int | Z-Score阈值，默认3 |
| get_scale | df_data | pd.DataFrame | 待标准化数据 |
| | x_columns | list | 特征列名，默认自动检测 |
| | y_column | list | 标签列名 |
| | scale_type | str | 'z-score'或'max-min' |
| assign_qbins | ds_data | pd.Series | 待分箱数据 |
| | quantiles | int | 分箱数量 |
        

## 返回值说明


- **normal_transform**: (转换后数据, lambda值)
- **normal_recovery**: 还原后的原始数据
- **polyfit**: (多项式对象, 残差标准差, 拟合值)
- **get_abnormal**: 异常数据序列
- **remove_abnormal**: 正常数据序列
- **get_scale**: (标准化后数据, 标准化列名列表)
- **get_transformer_position_encoding**: 位置编码矩阵
- **assign_qbins**: (分箱结果, 分箱区间)
        

## 注意事项


- Box-Cox变换要求数据必须大于0
- Z-Score检测基于正态分布假设
- 标准化会改变数据的尺度但不改变分布形状
- 多项式拟合阶数过高可能导致过拟合
- 分箱操作会损失数据的连续性信息
        

---
*生成时间: 2026-03-23 16:10:12*
*项目: FreeAeon-ML*
