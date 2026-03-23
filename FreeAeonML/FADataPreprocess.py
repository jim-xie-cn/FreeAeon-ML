'''
数据预处理
1. CFADataProcess (数据预处理)
   --将数据转换为正态分布（box-cox）
   --从正态分布中，将数据还原（box-cox）
   --获取异常数据（Z-Score）
2. CFATransformer(数据转换)
   --将原始数据转换另一组数据风格
   --scale: 线性缩放（不改变分布）
   --quantile: 改变分布，分位数映射（支持单变量或多变量）
   --copula:   改变分布，多变量copula变换（默认）
'''
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import norm
from scipy.special import inv_boxcox
from sklearn.preprocessing import StandardScaler
import os,sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from FreeAeonML.FASample import CFASample

class CFADataPreprocess:
    def __init__(self):
        pass
    '''
    将数据转换为正态分布（使用box-cox）
    lambda_value==0，使用log转换，lambda_value==None，自动计算最近lambda参数
    '''
    @staticmethod
    def normal_transform(ds_data,lambda_value=None):
        t_data, l_value = stats.boxcox(ds_data,lambda_value)
        return pd.Series(t_data,index=ds_data.index),l_value
    '''
    从正态分布中，将数据还原（使用box-cox）
    '''
    @staticmethod
    def normal_recovery(ds_data,lambda_value):
        original_data = inv_boxcox(np.array(ds_data.values),lambda_value)
        return pd.Series(original_data,index=ds_data.index)
    '''
    多项式拟合
    ds_x：pd.Series,自变量，当为空时，时间序列数据拟合。
    ds_y：pd.Series,因变量
    degree：多项式最高阶数
    返回值：拟合对象(p_object),拟合残差的标准差(e_std),拟合后的数据(v_object)
    '''
    def polyfit(ds_y,ds_x = pd.Series([]), degree = 2):
        if ds_x.empty :
            x = np.arange(ds_y.shape[0]) # 生成对应的时间点作为（自变量）
        else:
            x = ds_x.values()
        y = ds_y.values
    
        p_coefficients, e_residuals, _, _, _  = np.polyfit(x, y, degree,full=True)
    
        p_object = np.poly1d(p_coefficients)
        v_object = pd.Series(p_object(x),index=ds_y.index)
        e_std = np.sqrt(e_residuals / len(x))
        return p_object,e_std,v_object
    '''
    获取异常数据（Z-Score绝对值大于3 sigma）
    ''' 
    @staticmethod
    def get_abnormal(ds_data,n_sigma=3):
        ds_z_score = (ds_data - ds_data.mean()) / ds_data.std()
        filtered =  ds_z_score[(ds_z_score<(0-n_sigma) ) | (ds_z_score>n_sigma)]
        return ds_data.iloc[filtered.index]
    '''
    去掉异常数据（Z-Score绝对值大于3 sigma）
    ''' 
    @staticmethod
    def remove_abnormal(ds_data,n_sigma=3):
        ds_z_score = (ds_data - ds_data.mean()) / ds_data.std()
        filtered =  ds_z_score[(ds_z_score>(0-n_sigma) ) & (ds_z_score < n_sigma)]
        return ds_data.iloc[filtered.index]
        
    # 对特征字段进行标准化处理(z-score/max-min)
    @staticmethod
    def get_scale(df_data,x_columns=[],y_column=['label'],scale_type='z-score'):
        df_ret = df_data.copy(deep=True)
        if x_columns:
            scale_colums = x_columns
        else:
            scale_colums = []
            for key,type in zip(df_data.keys(),df_data.dtypes):
                if not type in ["bool","object","category",'datetime64','datetime'] and not key in y_column:
                    scale_colums.append(key)
                    if scale_type == 'z-score':
                        df_ret[[key]] = StandardScaler().fit_transform(df_ret[[key]])
                    elif scale_type == 'max-min':
                        df_ret[key] = (df_ret[key]-df_ret[key].min())/(df_ret[key].max()-df_ret[key].min())

                if type in ['bool']:
                    df_ret[key] = df_ret[key].astype(int)
                    
        #if not scale_colums:
        #    return df_ret,scale_colums
        #
        #if scale_type == 'z-score':
        #    scaler = StandardScaler()
        #    df_ret[scale_colums] = scaler.fit_transform(df_ret[scale_colums])
        #elif scale_type == 'max-min':
        #    for key in scale_colums:
        #        df_ret[key] = (df_ret[key]-df_ret[key].min())/(df_ret[key].max()-df_ret[key].min())
        return df_ret,scale_colums
        
    # 位置编码函数
    @staticmethod
    def get_transformer_position_encoding(seq_length, d_model):
        position_enc = np.array([
            [pos / np.power(10000, 2 * (i // 2) / d_model) for i in range(d_model)]
            for pos in range(seq_length)
        ])
        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # 偶数位置
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # 奇数位置
        return position_enc
        
    # 等频率划分
    @staticmethod
    def assign_qbins(ds_data , quantiles ):
        ds_raw = ds_data.copy(deep = True)
        ds_noised = ds_data.copy(deep = True)
        scale = 10
        ds_scaled = ds_raw * scale
        epsilon = 1e-8
        ds_noised = ds_scaled + np.random.uniform(epsilon, epsilon*2, size=len(ds_scaled))
        ds_noised[ds_noised.idxmin()] = ds_scaled.min()

        ds_noised = ds_noised.drop_duplicates(keep='first')
        labels=range(quantiles)
        _, bin_edges  = pd.qcut(ds_noised,q=quantiles,duplicates='drop',retbins=True,precision=8)
        if len(labels) >= len(bin_edges):
            labels = labels[0:(len(labels)-1)]
        bins,bin_edges = pd.cut(ds_scaled, bins=bin_edges, labels=labels, include_lowest=True,retbins=True)
        bin_edges = bin_edges / scale
        intervals = [[bin_edges[i], bin_edges[i + 1]] for i in range(len(bin_edges) - 1)]
        return bins,intervals

class CFATransformer:
    
    def __init__(self, mode='copula', method='average', adjust_corr=False):
        """
        mode:
            'scale'     → 线性缩放（不改变分布）
            'quantile'  → 分位数映射（单变量/多变量）
            'copula'    → 多变量 copula 变换（默认）
        """
        self.mode = mode
        self.method = method
        self.adjust_corr = adjust_corr
        
        # 存储参数（用于 inverse）
        self.params = {}
    
    # ---------- 基础函数 ----------
    
    @staticmethod
    def _empirical_cdf(series, method='average'):
        ranks = series.rank(method=method)
        return (ranks - 0.5) / len(series)
    
    @staticmethod
    def _quantile_map(u, target):
        target_sorted = np.sort(target)
        return np.interp(
            u,
            np.linspace(0, 1, len(target_sorted)),
            target_sorted
        )
    
    # ---------- 核心接口 ----------
    
    def fit(self, source, target):
        """学习映射关系"""
        A = source
        B = target
        self.is_series = isinstance(A, pd.Series)
        
        if self.is_series:
            A = A.to_frame("col")
            B = B.to_frame("col")
        
        self.columns = A.columns
        
        # ---------- scale ----------
        if self.mode == 'scale':
            self.params['mean_A'] = A.mean()
            self.params['std_A'] = A.std()
            self.params['mean_B'] = B.mean()
            self.params['std_B'] = B.std()
        
        # ---------- quantile ----------
        elif self.mode == 'quantile':
            self.params['B_sorted'] = {
                col: np.sort(B[col].values) for col in self.columns
            }
        
        # ---------- copula ----------
        elif self.mode == 'copula':
            
            # B 的边际分布
            self.params['B_sorted'] = {
                col: np.sort(B[col].values) for col in self.columns
            }
            
            if self.adjust_corr:
                # B 的协方差
                B_u = B.apply(lambda x: self._empirical_cdf(x, self.method))
                B_z = pd.DataFrame(norm.ppf(B_u), columns=B.columns)
                
                cov_B = np.cov(B_z.T)
                cov_B += np.eye(len(cov_B)) * 1e-6
                
                self.params['L_B'] = np.linalg.cholesky(cov_B)
        
        return self
    
    # ---------- transform ----------
    
    def transform(self, data):
        A = data
        if self.is_series:
            A = A.to_frame("col")
        
        A_out = pd.DataFrame(index=A.index, columns=A.columns)
        
        # ---------- scale ----------
        if self.mode == 'scale':
            A_out = (A - self.params['mean_A']) / self.params['std_A']
            A_out = A_out * self.params['std_B'] + self.params['mean_B']
        
        # ---------- quantile ----------
        elif self.mode == 'quantile':
            for col in self.columns:
                u = self._empirical_cdf(A[col], self.method)
                A_out[col] = self._quantile_map(u, self.params['B_sorted'][col])
        
        # ---------- copula ----------
        elif self.mode == 'copula':
            
            # Step1: uniform
            A_u = A.apply(lambda x: self._empirical_cdf(x, self.method))
            
            # Step2: normal
            A_z = pd.DataFrame(norm.ppf(A_u), columns=A.columns)
            
            # Step3: adjust corr
            if self.adjust_corr:
                cov_A = np.cov(A_z.T)
                cov_A += np.eye(len(cov_A)) * 1e-6
                
                L_A = np.linalg.cholesky(cov_A)
                A_z = A_z @ np.linalg.inv(L_A).T @ self.params['L_B'].T
            
            # Step4: back to uniform
            A_u_new = pd.DataFrame(norm.cdf(A_z), columns=A.columns)
            
            # Step5: map to B
            for col in self.columns:
                A_out[col] = self._quantile_map(
                    A_u_new[col], self.params['B_sorted'][col]
                )
        
        if self.is_series:
            return A_out.iloc[:, 0]
        
        return A_out.astype(float)
    
    # ---------- inverse ----------
    
    def inverse(self, data_transformed, data_original):
        A_transformed = data_transformed
        A_original = data_original
        if self.is_series:
            A_transformed = A_transformed.to_frame("col")
            A_original = A_original.to_frame("col")

        A_rec = pd.DataFrame(index=A_transformed.index, columns=A_transformed.columns)
        
        # ---------- scale ----------
        if self.mode == 'scale':
            A_rec = (A_transformed - self.params['mean_B']) / self.params['std_B']
            A_rec = A_rec * self.params['std_A'] + self.params['mean_A']
        
        # ---------- quantile / copula ----------
        else:
            for col in self.columns:
                ranks = A_transformed[col].rank(method=self.method)
                u = (ranks - 0.5) / len(ranks)
                
                A_rec[col] = self._quantile_map(
                    u, A_original[col].values
                )
        
        if self.is_series:
            return A_rec.iloc[:, 0]
        
        return A_rec.astype(float)
    
def main():
    #正态转换
    ds_data = pd.Series([1,2,3,4,5,6])
    transformed_data = CFADataPreprocess.get_abnormal(ds_data)
    ds_data = ds_data.drop(transformed_data.index)
    #ds_data = ds_data[ds_data>0]
    ds_data.plot()
    plt.show()    

    #正态拟合
    p_object,e_std,v_object = CFADataPreprocess.polyfit(ds_y=ds_data,degree=3)
    print(v_object)
    
    #标准化
    df_sample = CFASample.get_random_classification(1000,n_feature=10,n_class=2)
    df_scale,scale_colums = CFADataPreprocess.get_scale(df_sample,y_column='y',scale_type="max-min")
    print(df_scale)

    #准备数据
    df_source = pd.DataFrame({'x': np.random.normal(0, 1, 100),'y': np.random.exponential(1, 100)})
    df_target = pd.DataFrame({'x': np.random.normal(10, 2, 200),'y': np.random.exponential(5, 200)})
    
    #单变量分布转换
    ds_source,ds_target = df_source['x'],df_target['x']
    df_result = pd.DataFrame()
    min_size = min(len(ds_source),len(ds_target))
    df_result['Source'] = ds_source.head(min_size)
    df_result['Target'] = ds_target.head(min_size)

    for mode in ['scale','quantile','copula']:
        transer = CFATransformer(mode=mode).fit(ds_source,ds_target)
        ds_tranformed = transer.transform(ds_source)
        ds_recovered = transer.inverse(ds_tranformed, ds_source)
        df_result[f"transfom_{mode}"] = ds_tranformed
        df_result[f"inverse_{mode}"] = ds_recovered
    
    print(df_result)

    #多变量分布转换
    min_size = min(len(df_source),len(df_target))
    df_result = pd.DataFrame()
    for key in df_source:
        df_result[f"source_{key}"] = df_source.head(min_size)[key]
        df_result[f"target_{key}"] = df_target.head(min_size)[key]

    for mode in ['scale','quantile','copula']:
        transer = CFATransformer(mode=mode).fit(df_source,df_target)
        df_tranformed = transer.transform(df_source)
        df_recovered = transer.inverse(df_tranformed, df_source)
        for key in df_tranformed:
            df_result[f"transfom_{mode}_{key}"] = df_tranformed[key]
            df_result[f"inverse_{mode}_{key}"] = df_recovered[key]
    print(df_result)

if __name__ == "__main__":         
    main()                         
