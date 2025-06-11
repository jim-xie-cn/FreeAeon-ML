import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json,os,sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from FreeAeonML.FADataPreprocess import CFADataPreprocess
from FreeAeonML.FASample import CFASample

def main():
    ds_data = pd.Series([1,2,3,4,5,6])
    transformed_data = CFADataPreprocess.get_abnormal(ds_data)
    ds_data = ds_data.drop(transformed_data.index)
    #ds_data = ds_data[ds_data>0]
    ds_data.plot()
    plt.show()    
    
    p_object,e_std,v_object = CFADataPreprocess.polyfit(ds_y=ds_data,degree=3)
    print(v_object)
    
    df_sample = CFASample.get_random_classification(1000,n_feature=10,n_class=2)
    df_scale,scale_colums = CFADataPreprocess.get_scale(df_sample,y_column='y',scale_type="max-min")
    print(df_scale)

if __name__ == "__main__":
    main()