import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json,os,sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from FreeAeonML.FAFeatureSelect import CFAFeatureSelect
from FreeAeonML.FASample import CFASample
from FreeAeonML.FADataPreprocess import CFADataPreprocess
from FreeAeonML.FAModelSeries import CFAArima

def main():
    #准备训练数据
    ds_data = pd.Series([1,2,3,4,5,6])
    transformed_data = CFADataPreprocess.get_abnormal(ds_data)
    ds_data = ds_data.drop(transformed_data.index)
    ds_data = ds_data[ds_data>0]

    #设置季节性的参数范围
    seasonal_order_range = ((1,4),(1,4),(3,10),(5,10))

    #训练模型，找到最佳结果
    model,ret_order = CFAArima.auto_fit(ds_train = ds_data.head(4),
                                        ds_test = ds_data.tail(2),
                                        seasonal_order_range = seasonal_order_range)

    #显示模型
    CFAArima.show_model(model)
    predict_data = model.predict()

    #显示预测结果
    CFAArima.show_result(ds_data,predict_data)

    #周期性，季节性和趋势性分解
    decomposition = CFAArima.decomposition(ds_data,period=2)
    decomposition.plot()
    plt.show()

if __name__ == "__main__":
    main()
