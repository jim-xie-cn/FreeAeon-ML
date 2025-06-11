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
    
def main():
  ds = pd.Series([1,1,1,1,2,3,3,3,3,3,5])
  print("Changed node index ",CFACommon.find_changed_index(ds))
    
if __name__ == "__main__":
    main()



