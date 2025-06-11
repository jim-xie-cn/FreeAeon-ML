import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json,os,sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from FreeAeonML.FASample import CFASample

def main():
    np.random.seed(0)
    #生成分类数据，10个特征，4个类别
    df_classification = CFASample.get_random_classification(1000,n_feature=10,n_class=4)
    print(df_classification)

    #生成回归数据
    df_regression = CFASample.get_random_regression()
    print(df_regression)

    #生成聚类数据
    df_cluster = CFASample.get_random_cluster()
    print(df_cluster)

    #分割数据集
    df_train,df_test = CFASample.split_dataset(df_regression)
    print(df_train.shape)
    print(df_test.shape)

    #SMOTE平衡采样
    df_sample = CFASample.resample_smote(df_classification,y_column='y')
    print(df_sample)

    #普通平衡采样
    df_sample = CFASample.resample_balance(df_classification,y_column='y')
    print(df_sample)

    #找到发生变化的样本
    ds = pd.Series([1,1,1,1,2,3,3,3,3,3,5])
    print("Changed node index ",CFASample.find_changed_index(ds))

    #找到发生变化的样本
    ds = pd.Series([1,1,1,1,2,3,3,3,3,3,5])
    print("Changed node index ",CFASample.find_changed_index(ds))

    #DataFrame划分成多个batch
    df_sample = CFASample.get_random_classification(1000,n_feature=10,n_class=4)
    dfs = CFASample.split_dataframe(df_sample,batch_size = 10)
    print(dfs)

if __name__ == "__main__":
    main()