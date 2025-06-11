import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json,os,sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from FreeAeonML.FAFeatureSelect import CFAFeatureSelect
from FreeAeonML.FASample import CFASample
from FreeAeonML.FAModelClassify import CFAModelClassify
import h2o
from h2o.estimators import *

def main():
    #初始化
    np.random.seed(0)
    h2o.init(nthreads = -1, verbose=False)

    # 准备训练样本
    df_sample = CFASample.get_random_classification(1000,n_feature=1,n_class=2)
    df_train,df_test = CFASample.split_dataset(df_sample)
    print(df_train)
    #训练模型
    models = {
        "dt":H2ODecisionTreeEstimator(),
        "svm":H2OSupportVectorMachineEstimator(), 
        "rf":H2ORandomForestEstimator(),
        "ann":H2ODeepLearningEstimator(hidden=[100, 100]),
        "bayes":H2ONaiveBayesEstimator(),
        "glm":H2OGeneralizedLinearEstimator(),
        "gbm":H2OGradientBoostingEstimator(),
        #"xgboost":H2OXGBoostEstimator()
    }
    model_1 = CFAModelClassify(models = models)
    model_1.train(df_train,y_column='y')
    
    #保存模型
    model_1.save("./test")
    
    #读取模型
    model_2 = CFAModelClassify(models = models )
    model_2.load("./test")

    #使用模型进行预测
    df_pred = model_2.predict(df_test,y_column='y')
    print(df_pred)

    #评估模型性能
    df_evaluate = model_2.evaluate(df_test,y_column='y')
    print(df_evaluate)

    #获取特征重要程度
    df_importance = model_2.importance()
    print(df_importance)
    nan_rows = df_pred[df_pred.isna().any(axis=1)]
    print(nan_rows)
    df_pred = df_pred.dropna()
    print(df_pred)

if __name__ == "__main__":
    main()