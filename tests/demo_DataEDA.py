import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json,os,sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from FreeAeonML.FADataEDA import CFADataDistribution,CFADataTest,CFAFitter

def test_normal_test():
    #生成测试数据
    X_Normal = np.random.randn(1000)
    X_linear = np.linspace(1,100,1000)
    df_data = pd.DataFrame()
    df_data['Normal'] = pd.Series(X_Normal)
    df_data['Linear'] = pd.Series(X_linear)
    #正态性测试
    test = CFADataDistribution.normal_test(df_data['Normal'])
    print(json.dumps(test,indent=4))
    test = CFADataDistribution.dist_test(df_data['Normal'],df_data['Normal'])
    print(json.dumps(test,indent=4))
    #可视化结果
    CFADataDistribution.show_dist(df_data['Normal'],bins = 20)

def test_normal_fit():
    ds = np.random.normal(loc=0, scale=1, size=100)
    CFADataDistribution.normal_fit(ds)

def test_dist_test():
    ds1 = np.random.normal(loc=0, scale=1, size=100)
    ds2 = np.random.normal(loc=0, scale=1, size=100)
    result = CFADataDistribution.dist_test(ds1, ds2)
    print(result)
 
def test_stationarity_test():
    df_data = pd.DataFrame()
    df_data['Normal'] = pd.Series(np.random.randn(1000))
    df_test = CFADataTest.stationarity_test(df_data.Normal)
    print(df_test)
    CFADataTest.show_acf_pacf(df_data.Normal)
    plt.show()

def test_granger_test():
    # 构造样本数据（一个变量滞后影响另一个）
    n = 100
    x = np.random.normal(0, 1, n)
    y = np.roll(x, 1) + np.random.normal(0, 0.1, n)  # y受x滞后影响
    # 调用函数
    min_p, best_lag, approve_list, detail = CFADataTest.granger_test(ds_result=y, ds_source=x, maxlag=5, p_value=0.05)
    # 输出结果
    print(f"最小p值: {min_p}")
    print(f"最佳滞后期: {best_lag}")
    print("通过显著性检验的滞后期:")
    for item in approve_list:
        print(item)

def test_fitter():
    #数据拟合
    np.random.seed(0)
    x_data = np.linspace(0, 10, 100)
    y_data_linear = 2 * x_data + 1 + np.random.normal(size=x_data.size)
    y_data_polynomial = 1 * x_data**2 - 2 * x_data + 1 + np.random.normal(size=x_data.size)
    y_data_exponential = 2 * np.exp(0.5 * x_data) + np.random.normal(size=x_data.size)


    # 线性拟合
    linear_fitter = CFAFitter(model_type='linear')
    linear_params = linear_fitter.fit(x_data,y_data_linear)
    print("Linear fit parameters:", linear_params)
    linear_fitter.plot(x_data,y_data_linear)

    # 多项式拟合
    poly_fitter = CFAFitter(model_type='polynomial')
    poly_params = poly_fitter.fit(x_data,y_data_polynomial)
    print("Polynomial fit parameters:", poly_params)
    poly_fitter.plot(x_data,y_data_polynomial)

    # 指数拟合
    exp_fitter = CFAFitter(model_type="exponential")
    exp_params = exp_fitter.fit(x_data,y_data_exponential)
    print("Exponential fit parameters:", exp_params)
    exp_fitter.plot(x_data,y_data_exponential)

def main():
    np.random.seed(0)
    #是否为高斯分布
    test_normal_test()
    
    #正态拟合
    test_normal_fit()

    #检查两个分布是否相同
    test_dist_test()

    #平稳性检查
    test_stationarity_test()

    # 格兰特因果检验
    test_granger_test()

    # 数据拟合测试
    test_fitter()

if __name__ == "__main__":
    main()