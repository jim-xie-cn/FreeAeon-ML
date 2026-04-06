'''
特征选择
1. CFAFeatureSelect
   --生成信息图(分类数据)
   --GLM-ANOVA 方差检验（回归数据）
   --格兰特因果检验（时序数据）
'''
import h2o,json,os,sys,warnings
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from tqdm import tqdm
import os,sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from FreeAeonML.FACommon import CFACommon
from FreeAeonML.FASample import CFASample
from FreeAeonML.FACommon import CFACommon
from FreeAeonML.FAModelClassify import CFAModelClassify
import h2o
from h2o.estimators import *
from h2o.automl import H2OAutoML
from h2o.estimators import *
from statsmodels.tsa.stattools import grangercausalitytests
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MaxAbsScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.patches as mpatches
from matplotlib.cm import get_cmap
from sklearn.preprocessing import StandardScaler

np.set_printoptions(suppress=True)
pd.set_option('display.float_format',lambda x : '%.8f' % x)

class CFAFeatureSelect():
 
    def __del__(self):
       pass
       #h2o.shutdown()

    def __init__(self,ip='localhost',port=54321):
        #h2o.init(nthreads = -1, verbose=False)
        self.m_df_h2o = None
        self.m_col_x = []
        self.m_col_y = None

    def load(self,df_sample,x_columns = [],y_column='label',is_regression = False):
        
        if x_columns == []:
            total_columns = df_sample.keys().tolist()
        else:
            total_columns = x_columns
         
        if y_column in total_columns:
            total_columns.remove(y_column)
                
        self.m_col_x = total_columns
        self.m_col_y = y_column
        
        self.m_df_h2o = h2o.H2OFrame(df_sample)
        if not is_regression:
            self.m_df_h2o[self.m_col_y] = self.m_df_h2o[self.m_col_y].asfactor()
    '''
    横坐标：信息总量（total information）
        --变量对预测的影响，,即该变量与其他变量的相关性。
        --横轴上的值越大，表示变量对响应的影响越显著。
    纵坐标：净信息（net information）
        --变量的独特性，总信息量.
        --净信息越高，预测能力越强，表示该变量对响应的影响越独特。
    可接受特征
        --位于虚线以上和右侧的特征是最具预测能力和独特性的特征，它们被认为是可接受的特征，
        --这些特征是被认为是核心驱动因素的变量，它们在总信息（预测能力）和净信息（独特性）方面都表现出色。        
        --可以用于建立模型和做出决策。
    返回值方法：
    ig.get_admissible_score_frame()
    ig.get_admissible_features()
    '''
    def get_inform_graph(self,algorithm="AUTO",protected_columns=[]): #["All",'AUTO','deeplearning','drf','gbm','glm','xgboost']
        if algorithm == "All":
            ret = {}
            for algor in ['AUTO','deeplearning','drf','gbm','glm']: #,'xgboost']: xgboost 有 bug
                if protected_columns:
                    ig = H2OInfogram(algorithm=algor,protected_columns=protected_columns)
                else:
                    ig = H2OInfogram(algorithm=algor)
                ig.train(x=self.m_col_x, y=self.m_col_y,training_frame=self.m_df_h2o)
                ret[algor] = ig
            return ret
        else:
            if protected_columns:
                ig = H2OInfogram(algorithm=algorithm,protected_columns=protected_columns)
            else:
                ig = H2OInfogram(algorithm=algorithm)
            ig.train(x=self.m_col_x, y=self.m_col_y,training_frame=self.m_df_h2o)
            return ig
    '''
    统计自变量和因变量的相关性
        --p值小于0.05,认为有相关性
        --通过特征组集合，查看特征之间是否有相关性
    '''           
    def get_anovaglm(self,family='gaussian',lambda_ = 0,highest_interaction_term=2):
        anova_model = H2OANOVAGLMEstimator(family=family,
                                   lambda_=lambda_,
                                   missing_values_handling="skip",
                                   highest_interaction_term=highest_interaction_term)
        anova_model.train(x=self.m_col_x, y=self.m_col_y, training_frame=self.m_df_h2o)
        return anova_model
        #anova_model.summary()

    '''
    格兰特因果检验
    ds_result：结果序列（必须为平稳序列）
    ds_source：原因序列（必须为平稳序列）
    maxlag：时间间隔（整数或列表，为整数时，遍历所有的lag）
    返回值：
    1. 最小的p值
    2. 最佳lag
    3. approve_list通过测试的lag
    4. 详细测试结果
    '''
    @staticmethod
    def granger_test(ds_result,ds_source,maxlag,p_value = 0.05):

        df_test = pd.DataFrame()
        df_test['result'] = ds_result
        df_test['source'] = ds_source

        gc_result = grangercausalitytests(df_test[['result', 'source']], maxlag=maxlag,verbose=False )
        min_lag_p = 100
        best_lag = -1
        detail = {}
        approve_list = []
        for lag in gc_result:

            result = gc_result[lag][0]
            detail[lag] = result.copy()
    
            max_p = 0 
            for test in result:
                test_value = result[test][0]
                test_p = result[test][1]
                if test_p > max_p:
                    max_p = test_p

            if max_p > p_value:
                continue
            #if max_p == 0:
            #    continue   
            approve_list.append({'lag':lag,"max p-value":max_p})
            
            if min_lag_p > max_p:
                min_lag_p = max_p
                best_lag = lag 

        return min_lag_p, best_lag, approve_list, detail   
    
    '''
    使用PCA，得到降维后和重构后的矩阵（根据特征值和特征向量）
    dataMat：原始矩阵
    n：特征向量的维度（降维后）
    '''
    @staticmethod
    def get_matrix_by_pca(dataMat, n):
        # 零均值化
        def zeroMean(dataMat):
            meanVal = np.mean(dataMat, axis=0)  # 按列求均值，即求各个特征的均值
            newData = dataMat - meanVal
            return newData, meanVal

        newData, meanVal = zeroMean(dataMat)
        covMat = np.cov(newData, rowvar=0)  # 求协方差矩阵,return ndarray；若rowvar非0，一列代表一个样本，为0，一行代表一个样本
        eigVals, eigVects = np.linalg.eig(np.asmatrix(covMat))  # 求特征值和特征向量,特征向量是按列放的，即一列代表一个特征向量
        # argsort将x中的元素从小到大排列，提取其对应的index(索引)
        eigValIndice = np.argsort(eigVals)  # 对特征值从小到大排序
        # print(eigValIndice)
        n_eigValIndice = eigValIndice[-1:-(n + 1):-1]  # 最大的n个特征值的下标
        n_eigVect = eigVects[:, n_eigValIndice]  # 最大的n个特征值对应的特征向量
        lowDDataMat = newData * n_eigVect  # 低维特征空间的数据
        reconMat = (lowDDataMat * n_eigVect.T) + meanVal  # 重构数据
        return lowDDataMat, reconMat

    # PCA降维度
    @staticmethod
    def get_pca(df, group='y', n_components=2, sample=None,feature_cols=[]):
        if group not in df.columns:
            raise KeyError(f'列 {group} 不在 DataFrame 中')
        if sample is not None and sample < len(df):
            df_vis = df.groupby(group, group_keys=False).apply(lambda x: x.sample(min(len(x), sample), random_state=0),include_groups=True)
        else:
            df_vis = df.copy()
            
        if not feature_cols :
            feature_cols = df.select_dtypes(include='number').columns.tolist()
        if group in feature_cols:
            feature_cols.remove(group)
        if len(feature_cols) <= 0:
            raise ValueError('没有可用于 PCA 的数值型特征列')
            
        X = df_vis[feature_cols].to_numpy()
        if X.shape[0] < n_components:
            raise ValueError(f'样本量只有 {X.shape[0]}，无法计算 {n_components} 个主成分')
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        pca = PCA(n_components=n_components, random_state=0)
        pca_results = pca.fit_transform(X_scaled)
        for i in range(n_components):
            df_vis[f'pca_{i}'] = pca_results[:, i]
        return df_vis
    
    # SVD降维
    @staticmethod
    def get_svd(df, group="y", n_components=2,sample=None,feature_cols=[]):
        if group not in df.columns:
            raise KeyError(f'列 {group} 不在 DataFrame 中')
        if sample is not None and sample < len(df):
            df_vis = df.groupby(group, group_keys=False).apply(lambda x: x.sample(min(len(x), sample), random_state=0),include_groups=True)
        else:
            df_vis = df.copy()

        if not feature_cols :
            feature_cols = df.select_dtypes(include='number').columns.tolist()
        if group in feature_cols:
            feature_cols.remove(group)
        if len(feature_cols) <= 0:
            raise ValueError('没有可用于 PCA 的数值型特征列')
        
        X = df_vis[feature_cols]
        if not hasattr(X, "todense"):
            scaler = MaxAbsScaler()
            X_scaled = scaler.fit_transform(X)
        else:
            X_scaled = X  # 已经是稀疏矩阵

        svd = TruncatedSVD(n_components=n_components, random_state=0)
        coords = svd.fit_transform(X_scaled)
        for i in range(n_components):
            df_vis[f'svd_{i}'] = coords[:, i]
        return df_vis

    # t-SNE降维
    @staticmethod
    def get_t_sne(df,group="y", n_components=2,sample=None,feature_cols=[],scale=True,use_pca=True,n_pca=50,perplexity=30,learning_rate=200,n_iter=1500):
        if group not in df.columns:
            raise KeyError(f'列 {group} 不在 DataFrame 中')
        if sample is not None and sample < len(df):
            df_vis = df.groupby(group, group_keys=False).apply(lambda x: x.sample(min(len(x), sample), random_state=0),include_groups=True)
        else:
            df_vis = df.copy()
            
        if not feature_cols :
            feature_cols = df.select_dtypes(include='number').columns.tolist()
        if group in feature_cols:
            feature_cols.remove(group)
        if len(feature_cols) <= 0:
            raise ValueError('没有可用于 PCA 的数值型特征列')
            
        X = df_vis[feature_cols]
        
        if scale:
            scaler = StandardScaler(with_mean=False)  # 如果数据不是稀疏/非负，可以改成默认
            X = scaler.fit_transform(X)

        if use_pca and X.shape[1] > n_pca:
            pca = PCA(n_components=n_pca, random_state=0)
            X = pca.fit_transform(X)

        tsne = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            learning_rate=learning_rate,
            n_iter=n_iter,
            init="pca",
            random_state=0,
            early_exaggeration=24,
            metric="cosine"  # or euclidean
        )
        tsne_results = tsne.fit_transform(X)
        for i in range(n_components):
            df_vis[f'tsne_{i}'] = tsne_results[:, i]

        return df_vis
    
    # 显示降维结果
    @staticmethod
    def show_components(df_data,group="y",ax = None):
        pca_components = [col for col in ('pca_0', 'pca_1', 'pca_2') if col in df_data]
        svd_components = [col for col in ('svd_0', 'svd_1', 'svd_2') if col in df_data]
        tsne_components = [col for col in ('tsne_0', 'tsne_1', 'tsne_2') if col in df_data]
        cmap = get_cmap('tab10')
        group_codes = df_data[group].astype('category').cat.codes 
        group_categories = df_data[group].astype('category').cat.categories
        palette = {label: cmap(i) for i, label in enumerate(group_categories)}
        if len(pca_components) > 0:
            target_components = pca_components
            title = "PCA components"
        elif len(svd_components) > 0:
            target_components = svd_components
            title = "SVD components"
        elif len(tsne_components) > 0:
            target_components = tsne_components
            title = "t-SNE components"
        else:
            print("only support PCA,SVD and t-SNE")
            return
        if target_components:
            if len(target_components) == 2:
                if ax is None:
                    fig, ax = plt.subplots(figsize=(6, 5))

                sns.scatterplot(data=df_data,
                                x=target_components[0],
                                y=target_components[1],
                                hue=group,
                                palette=palette,
                                s=20,
                                alpha=0.6,
                                edgecolor='none',
                                ax=ax,
                                legend=False)
                handles = [mpatches.Patch(color=palette[label], label=label)
                        for label in group_categories]
                ax.legend(handles=handles, title='group', loc='best')
                ax.set_xlabel(target_components[0])
                ax.set_ylabel(target_components[1])
                ax.set_title(title)
            else:
                if ax is None:
                    fig = plt.figure(figsize=(6, 5))
                    ax3d = fig.add_subplot(111, projection='3d')
                else:
                    fig = ax.figure
                    spec = ax.get_subplotspec()
                    ax.remove()
                    ax3d = fig.add_subplot(spec, projection='3d')
            
                cmap_obj = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
                sc = ax3d.scatter(df_data[target_components[0]],
                                df_data[target_components[1]],
                                df_data[target_components[2]],
                                c=group_codes,
                                cmap=cmap_obj,
                                s=20,
                                alpha=0.6,
                                edgecolor='none')
            
                # 3D 图 legend
                mask = ~np.isnan(group_codes)
                valid_codes = np.unique(group_codes[mask].astype(int))
                categories = list(group_categories)
            
                if valid_codes.size:
                    code_offset = 1 if valid_codes.min() == 1 and len(valid_codes) == len(categories) else 0
                    bounds = np.arange(valid_codes.min(), valid_codes.max() + 2) - 0.5
                    norm = mcolors.BoundaryNorm(bounds, cmap_obj.N)
                    sm = cm.ScalarMappable(norm=norm, cmap=cmap_obj)
            
                    handles = []
                    for code in valid_codes:
                        idx = int(code - code_offset)
                        if 0 <= idx < len(categories):
                            label = categories[idx]
                        else:
                            label = str(code)
                        handles.append(mpatches.Patch(color=sm.to_rgba(code), label=label))
            
                    ax3d.legend(handles=handles, title='group', loc='best')
            
                ax3d.set_xlabel(target_components[0].upper())
                ax3d.set_ylabel(target_components[1].upper())
                ax3d.set_zlabel(target_components[2].upper())
                ax3d.set_title(title)

            plt.tight_layout()
    
    '''
    初步验证区分度
    '''
    @staticmethod
    def mode_train_test(df_data,group="y"):
        h2o.init(nthreads = -1, verbose=False)
        models = {
            "rf":H2ORandomForestEstimator(),
            "xgboost":H2OXGBoostEstimator(),
            "ann":H2ODeepLearningEstimator()
        }
        df_sample = df_data.copy(deep = True)
        if pd.api.types.is_string_dtype(df_sample[group]) or df_sample[group].dtype == "object":
            df_sample[group] = df_sample[group].astype("category")
            df_sample[group] = df_sample[group].cat.codes.astype("int32")  

        df_train,df_test = CFASample.split_dataset(df_sample)
        model = CFAModelClassify(models = models)
        model.train(df_train,y_column=group)
        df_evaluate = model.evaluate(df_test,y_column=group)
        return df_evaluate

def main():
    
    #如果是WSL,注释掉h2o.init(),使用h2o.connect()
    h2o.init(nthreads = -1, verbose=False)
    #h2o.connect(ip=ip,port=port)

    #分类数据,查看信息图
    df_sample = CFASample.get_random_classification(1000,n_feature=10,n_class=2)
    Fea = CFAFeatureSelect()
    Fea.load(df_sample,x_columns = ['x1','x2'],y_column='y',is_regression = False)
    ig = Fea.get_inform_graph("AUTO")
    ig.plot()
    ig.show()

    #回归数据，查看方差检验
    df_sample = CFASample.get_random_regression()
    Fea.load(df_sample,x_columns = ['x1','x2'],y_column='y',is_regression = True)
    ag = Fea.get_anovaglm()
    print(ag.summary())

    #格兰特因果检验
    a = [1,-1,2,-2,3,-3.1]
    b = [2,-2,3,-3,4,-4.1]
    result = CFAFeatureSelect.granger_test(b,a,[1])
    print(result)
        
    a.extend(a)
    b.extend(b)
    result = CFAFeatureSelect.granger_test(b,a,2)
    print(result)

    #降维
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

    #使用模型初步验证
    df_result = CFAFeatureSelect.mode_train_test(df_sample)
    print(df_result)

    # 降维到2维
    data = np.random.rand(500, 5)
    labels = np.random.randint(0, 2, size=500)
    df_data = pd.DataFrame(data)
    df_data['y'] = labels
    lowDData, reconData = CFAFeatureSelect.get_matrix_by_pca(data, 2)

    print("降维后的数据形状:", lowDData.shape)  # (10, 2)
    print("重构数据形状:", reconData.shape)      # (10, 5)

    print("降维后的数据:\n", lowDData)
    print("重构后的数据:\n", reconData)

if __name__ == "__main__":
    main()
