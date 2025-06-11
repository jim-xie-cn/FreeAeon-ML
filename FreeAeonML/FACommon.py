#!/usr/bin/env python
# coding: utf-8
#!/usr/bin/env bash
import pandas as pd
import numpy as np
from tqdm import tqdm
import subprocess

class CFACommon:
    def __init__(self):
        pass
    '''
    带进度条，读取csv文件
    '''
    @staticmethod
    def load_csv(file_name,chunksize=1000):
        out = subprocess.getoutput("wc -l %s" % file_name)
        total = int(out.split()[0]) / chunksize
        return pd.concat([chunk for chunk in tqdm(pd.read_csv(file_name, chunksize=chunksize),total=total, desc='Loading data %s'%file_name)])
    
    '''
    找到序列中，发生变化的index
    '''
    @staticmethod
    def find_changed_index(ds):
        df = pd.DataFrame()
        ds_1 = ds.fillna(1e-11)  # 用极小值填充 NaN
        ds_1 = pd.concat([ds_1, pd.Series([ds.iloc[-1]], index=[ds.index[-1] + 1])])
        ds_2 = ds_1.shift(1)
        df['x'] = ds_1
        df['y'] = ds_2
        df=df[0:-1]
        return df[df['x']!=df['y']].index.values
    
    '''
    将DataFrame划分成多个batch
    '''
    @staticmethod
    def split_dataframe(df_sample,batch_size = 10):
        return [df_sample[i:i + batch_size] for i in range(0, len(df_sample), batch_size)]
    
def main():
  ds = pd.Series([1,1,1,1,2,3,3,3,3,3,5])
  print("Changed node index ",CFACommon.find_changed_index(ds))
    
if __name__ == "__main__":
    main()



