import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json,os,sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from FreeAeonML.FADataEDA import CFADataDistribution,CFADataTest

def main():
    #生成测试数据
    np.random.seed(11)
    X_Normal = np.random.randn(1000)
    X_linear = np.linspace(1,100,1000)
    df_data = pd.DataFrame()
    df_data['Normal'] = pd.Series(X_Normal)
    df_data['Linear'] = pd.Series(X_linear)

    #是否为高斯分布
    test = CFADataDistribution.normal_test(df_data['Normal'])
    print(json.dumps(test,indent=4))
    test = CFADataDistribution.dist_test(df_data['Normal'],df_data['Normal'])
    print(json.dumps(test,indent=4))
    CFADataDistribution.show_dist(df_data['Normal'],bins = 20)

    #平稳性检查
    df_test = CFADataTest.stationarity_test(df_data.Normal)
    print(df_test)
    CFADataTest.show_acf_pacf(df_data.Normal)
    plt.show()

if __name__ == "__main__":
    main()