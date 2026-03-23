import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json,os,sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from FreeAeonML.FADataPreprocess import CFADataPreprocess,CFATransformer
from FreeAeonML.FASample import CFASample

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import inv_boxcox
from sklearn.preprocessing import StandardScaler

# 测试 normal_transform 函数
def test_normal_transform():
    """
    测试将数据转换为正态分布（Box-Cox变换），自动计算lambda值
    """
    data = pd.Series(np.random.exponential(scale=2.0, size=100), name='data')  # 非正态数据
    transformed, lambda_val = CFADataPreprocess.normal_transform(data)
    print("normal_transform:\n", transformed.head())
    print("lambda:", lambda_val)

# 测试 normal_recovery 函数
def test_normal_recovery():
    """
    测试从Box-Cox转换后的正态数据还原为原始数据
    """
    data = pd.Series(np.random.exponential(scale=2.0, size=100), name='data')
    transformed, lambda_val = CFADataPreprocess.normal_transform(data)
    recovered = CFADataPreprocess.normal_recovery(transformed, lambda_val)
    print("normal_recovery:\n", recovered.head())

# 测试 polyfit 函数
def test_polyfit():
    """
    测试多项式拟合函数，默认使用索引作为自变量
    """
    data_y = pd.Series(np.random.rand(100), name='y')  # 随机因变量
    p_obj, std_err, fitted = CFADataPreprocess.polyfit(data_y)
    print("polyfit coefficients:", p_obj.coefficients)
    print("std error:", std_err)
    print("fitted values:\n", fitted.head())

# 测试 get_abnormal 函数
def test_get_abnormal():
    """
    测试检测Z-score大于n_sigma的异常值
    """
    data = pd.Series(np.random.randn(100), name='z_test')  # 标准正态分布
    abnormal = CFADataPreprocess.get_abnormal(data, n_sigma=2)
    print("abnormal values:\n", abnormal)

# 测试 remove_abnormal 函数
def test_remove_abnormal():
    """
    测试移除Z-score大于n_sigma的异常值
    """
    data = pd.Series(np.random.randn(100), name='z_test')
    cleaned = CFADataPreprocess.remove_abnormal(data, n_sigma=2)
    print("cleaned data:\n", cleaned)

# 测试 get_scale 函数
def test_get_scale():
    """
    测试对DataFrame的特征列进行标准化处理（z-score）
    """
    df = pd.DataFrame({
        'feature1': np.random.randn(100),
        'feature2': np.random.rand(100),
        'label': np.random.randint(0, 2, 100)
    })
    scaled_df, cols = CFADataPreprocess.get_scale(df, scale_type='z-score')
    print("scaled columns:", cols)
    print("scaled df:\n", scaled_df.head())

# 测试 get_transformer_position_encoding 函数
def test_get_transformer_position_encoding():
    """
    测试生成Transformer模型的位置编码矩阵
    """
    encoding = CFADataPreprocess.get_transformer_position_encoding(seq_length=5, d_model=8)
    print("transformer position encoding:\n", encoding)

# 测试 assign_qbins 函数
def test_assign_qbins():
    """
    测试将数据等频率分箱，并返回箱编号与区间范围
    """
    data = pd.Series(np.random.randn(100), name='qcut_test')
    bins, intervals = CFADataPreprocess.assign_qbins(data, quantiles=4)
    print("binned labels:\n", bins.head())
    print("intervals:\n", intervals)

#测试数据转换
#1. 支持三种变换：scale,quantile,copula
#2. 支持单变量(pd.Series)和多变量(pd.DataFrame)转换
def test_transfomer():
    #准备数据
    df_source = pd.DataFrame({'x': np.random.normal(0, 1, 100),'y': np.random.exponential(1, 100)})
    df_target = pd.DataFrame({'x': np.random.normal(10, 2, 200),'y': np.random.exponential(5, 200)})
    #单变量分布转换
    ds_source,ds_target = df_source['x'],df_target['x']
    df_result = pd.DataFrame()
    min_size = min(len(ds_source),len(ds_target))
    df_result['Source'] = ds_source.head(min_size)
    df_result['Target'] = ds_target.head(min_size)

    for mode in ['scale','quantile','copula']:
        transer = CFATransformer(mode=mode).fit(ds_source,ds_target)
        ds_tranformed = transer.transform(ds_source)
        ds_recovered = transer.inverse(ds_tranformed, ds_source)
        df_result[f"transfom_{mode}"] = ds_tranformed
        df_result[f"inverse_{mode}"] = ds_recovered
    
    print(df_result)

    #多变量分布转换
    min_size = min(len(df_source),len(df_target))
    df_result = pd.DataFrame()
    for key in df_source:
        df_result[f"source_{key}"] = df_source.head(min_size)[key]
        df_result[f"target_{key}"] = df_target.head(min_size)[key]

    for mode in ['scale','quantile','copula']:
        transer = CFATransformer(mode=mode).fit(df_source,df_target)
        df_tranformed = transer.transform(df_source)
        df_recovered = transer.inverse(df_tranformed, df_source)
        for key in df_tranformed:
            df_result[f"transfom_{mode}_{key}"] = df_tranformed[key]
            df_result[f"inverse_{mode}_{key}"] = df_recovered[key]
    print(df_result)

# 执行所有测试函数
if __name__ == "__main__":
    test_normal_transform()
    test_normal_recovery()
    test_polyfit()
    test_get_abnormal()
    test_remove_abnormal()
    test_get_scale()
    test_get_transformer_position_encoding()
    test_assign_qbins()
    test_transfomer()
