import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json,os,sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from FreeAeonML.FAFeatureSelect import CFAFeatureSelect
from FreeAeonML.FASample import CFASample
from FreeAeonML.FACommon import CFACommon
import h2o

#信息图
def test_get_inform_graph():
    df_sample = CFASample.get_random_classification(1000,n_feature=10,n_class=2)
    Fea = CFAFeatureSelect()
    Fea.load(df_sample,x_columns = ['x1','x2'],y_column='y',is_regression = False)
    ig = Fea.get_inform_graph("AUTO")
    ig.plot()
    ig.show()

#方差分析
def test_get_anovaglm():
    df_sample = CFASample.get_random_regression()
    Fea = CFAFeatureSelect()
    Fea.load(df_sample,x_columns = ['x1','x2'],y_column='y',is_regression = True)
    ag = Fea.get_anovaglm()
    print(ag.summary())

#格兰特因果检验
def test_granger_test():
    a = [1,-1,2,-2,3,-3.1]
    b = [2,-2,3,-3,4,-4.1]
    result = CFAFeatureSelect.granger_test(b,a,[1])
    print(result)
        
    a.extend(a)
    b.extend(b)
    result = CFAFeatureSelect.granger_test(b,a,2)
    print(result)

#测试降维
def test_decomponse():
    df_sample = CFASample.get_random_classification(1000,n_feature=10,n_class=2)
    df_svd = CFAFeatureSelect.get_pca(df_sample,n_components=2,sample=None)
    df_pca = CFAFeatureSelect.get_svd(df_sample,n_components=2,feature_cols=['x0','x1','x2','x3'],sample=None)
    df_sne = CFAFeatureSelect.get_t_sne(df_sample,n_components=3,sample=100)
    fig, axes = plt.subplots(1, 3, figsize=(12, 6))
    axes = axes.flatten()
    CFAFeatureSelect.show_components(df_pca,ax=axes[0])
    CFAFeatureSelect.show_components(df_svd,ax=axes[1])
    CFAFeatureSelect.show_components(df_sne,ax=axes[2])
    plt.show()

#使用模型，初步验证
def test_model_train_test():
    df_sample = CFASample.get_random_classification(1000,n_feature=10,n_class=2)
    df_result = CFAFeatureSelect.mode_train_test(df_sample)
    print(df_result)

#图像降维与重构
def test_get_matrix_by_pca():
    # 生成示例数据 10个样本，5个特征
    data = np.random.rand(10, 5)

    # 降维到2维
    lowDData, reconData = CFAFeatureSelect.get_matrix_by_pca(data, 2)

    print("降维后的数据形状:", lowDData.shape)  # (10, 2)
    print("重构数据形状:", reconData.shape)      # (10, 5)

    print("降维后的数据:\n", lowDData)
    print("重构后的数据:\n", reconData)

def main():
    np.random.seed(0)

    #如果是WSL,注释掉h2o.init(),使用h2o.connect()
    h2o.init(nthreads = -1, verbose=False)
    #h2o.connect(ip=ip,port=port)

    #分类数据,查看信息图
    test_get_inform_graph()

    #回归数据，查看方差检验
    test_get_anovaglm()

    #格兰特因果检验
    test_granger_test()
    
    #降维
    test_decomponse()
    #使用模型验证
    test_model_train_test()

    #PCA矩阵降维(most for image)
    test_get_matrix_by_pca()

if __name__ == "__main__":
    main()
