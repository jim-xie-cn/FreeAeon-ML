import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json,os,sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from FreeAeonML.FAFeatureSelect import CFAFeatureSelect
from FreeAeonML.FASample import CFASample
from FreeAeonML.FAModelCluster import CFAModelCluster
from sklearn.cluster import KMeans,AffinityPropagation,AgglomerativeClustering,Birch,DBSCAN,MeanShift,OPTICS

def main():
    #初始化
    np.random.seed(0)
    
    #准备训练样本
    df_sample = CFASample.get_random_cluster()

    # 使用系统自带的模型进行训练
    models = None
    model = CFAModelCluster(cluster_count = 2, models = models)

    #如果需要使用指定的模型进行训练，请按照以下格式指定模型）
    #models = { "KMeans": KMeans(n_clusters= 2, random_state=False),"AffinityPropagation":AffinityPropagation(damping=0.9)}
    #model = CFAModelCluster(cluster_count = 2, models = models)

    #进行聚类
    df_result = model.fit_predict(df_sample)
    print(df_result)

    #评估聚类效果
    df_perf = model.evaluate(df_sample)
    print(df_perf)

    #显示聚类后的样本
    df_cluster_sample = model.sample_cluster(df_result,df_sample,"KMeans")
    print(df_cluster_sample)

if __name__ == "__main__":
    main()
