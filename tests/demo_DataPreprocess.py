import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json,os,sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from FreeAeonML.FADataPreprocess import CFADataPreprocess
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
