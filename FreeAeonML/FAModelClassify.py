'''
自动化训练分类模型（支持多分类），包括
1.模型训练
2.模型评估
3.ROC曲线绘制
'''
import platform
import numpy as np
import pandas as pd
import h2o,json,os,sys,time,datetime,warnings
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from h2o.automl import H2OAutoML
from h2o.estimators import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_regression,make_classification
from sklearn.metrics import recall_score,confusion_matrix,matthews_corrcoef,accuracy_score,precision_score,roc_auc_score,log_loss,f1_score,roc_curve,fbeta_score
from sklearn.metrics import confusion_matrix
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from FreeAeonML.FACommon import CFACommon
from FreeAeonML.FASample import CFASample
np.set_printoptions(suppress=True)
pd.set_option('display.float_format',lambda x : '%.8f' % x)

'''
使用h2o进行automl
'''
class CFAModelClassify():
    
    def __init__(self,ip='localhost',port=54321,models = None):
        #h2o.init(nthreads = -1, verbose=False)
        #h2o.connect(ip=ip,port=port,verbose=False)
        if models == None:
            self.m_models = {
                "dt":H2ODecisionTreeEstimator(),
                "svm":H2OSupportVectorMachineEstimator(), # kernel='gaussian'
                "rf":H2ORandomForestEstimator(),
                "ann":H2ODeepLearningEstimator(hidden=[100, 100]),
                "bayes":H2ONaiveBayesEstimator(),
                "glm":H2OGeneralizedLinearEstimator(),
                "gbm":H2OGradientBoostingEstimator()
                #"xgboost":H2OXGBoostEstimator()
            }
            # 判断平台与可用性
            system = platform.system()
            machine = platform.machine()

            try:
                if H2OXGBoostEstimator.available():
                    if system != "Windows" and not machine.startswith("arm"):
                        self.m_models["xgboost"] = H2OXGBoostEstimator()
                    else:
                        print("⚠️ 当前平台不支持 H2O XGBoost（Windows 或 ARM 架构）")
                else:
                    print("⚠️ H2O XGBoostEstimator 不可用（编译未启用或未安装）")
            except Exception as e:
                print(f"⚠️ 检查 H2O XGBoost 可用性失败：{e}")

        else:
            self.m_models = models

    def train(self,df_sample,x_columns = [] ,y_column='label',train_ratio = 0.85):
        
        if x_columns == []:
            total_columns = df_sample.keys().tolist()
        else:
            total_columns = x_columns
            if not y_column in total_columns:
                total_columns.append(y_column)
            
        df_temp = df_sample[total_columns]
        
        df_h2o = h2o.H2OFrame(df_temp)
        if train_ratio > 0:
            df_train, df_valid = df_h2o.split_frame(ratios=[train_ratio], seed=1234)
        else:
            df_train = df_h2o
            
        x = df_train.columns
        y = y_column
        x.remove(y)
        df_train[y] = df_train[y].asfactor()
        if train_ratio > 0:
            df_valid[y] = df_valid[y].asfactor()
        
        for key in self.m_models:
            model = self.m_models[key]
            #print("begin train ",key)
            if train_ratio > 0:
                model.train(x=x, y=y,training_frame=df_train,validation_frame=df_valid)
            else:
                model.train(x=x, y=y,training_frame=df_train)
            #print("end train ",key)
    
    def load(self,model_path):
        for key in self.m_models:
            folder = "%s/%s"%(os.path.abspath(model_path),key)
            model = self.m_models[key]
            all_models = os.listdir(folder)
            all_models.sort()
            for file_name in all_models:
                model_file = os.path.join(folder, file_name)
                print("loading",model_file)
                self.m_models[key] = h2o.load_model(model_file)
   
    def save(self,model_path):
        for key in self.m_models:
            folder = "%s/%s"%(os.path.abspath(model_path),key)
            model = self.m_models[key]
            if not os.path.exists(folder):
                os.makedirs(folder)
                
            for file_name in os.listdir(folder):
                model_file = os.path.join(folder, file_name)
                os.remove(model_file)
            
            model_saved = h2o.save_model(model=model, path=folder, force=True)
            print("save model to ",model_saved)

    def predict(self,df_sample,x_columns = [],y_column='label'):
        
        if x_columns == []:
            total_columns = df_sample.keys().tolist()
        else:
            total_columns = x_columns
         
        if y_column in total_columns:
            total_columns.remove(y_column)
                
        if y_column in df_sample.keys().tolist():
            y_true = df_sample[y_column].tolist()
        else:
            y_true = None
            
        df_temp = df_sample[total_columns]
        df_h2o = h2o.H2OFrame(df_temp)

        result = []
        for key in self.m_models:
            model = self.m_models[key]
            training_columns = model._model_json['output']['names'][:-1]
            for col in training_columns:
                if col not in df_temp.columns:
                    print("NO training column %s found not"%col)
                    df_temp = df_temp.reindex(columns=training_columns, fill_value=0)  
                    df_h2o = h2o.H2OFrame(df_temp)
                    break
            df_t = model.predict(df_h2o).as_data_frame()
            df_t['model'] = key
            df_t['true'] = y_true
            result.extend(json.loads(df_t.to_json(orient='records')))
            
        return pd.DataFrame(result)
    
    #average is [None, 'micro', 'macro', 'weighted']
    def evaluate(self,df_sample,x_columns = [],y_column='label',average='weighted'):
        
        df_pred = self.predict( df_sample,x_columns=x_columns , y_column=y_column )
        prob_list = []
        for i in range( df_sample[y_column].nunique() ):
            prob_list.append("p%d"%i)  
        result = []
        for model,df in df_pred.groupby('model'):
            y_pred = df['predict'].tolist()
            y_true = df['true'].tolist()
            for key in prob_list:
                if not key in df.keys().to_list():
                    df[key] = 0
            y_prab = df[prob_list].to_numpy()
            tmp = {}
            tmp['model'] = model
            tmp['confusion_matrix'] = confusion_matrix(y_true,y_pred)
            tmp['recall'] = recall_score(y_true,y_pred,average=average)
            tmp['mcc'] = matthews_corrcoef(y_true,y_pred)
            tmp['accuracy'] = accuracy_score(y_true,y_pred)
            tmp['precision'] = precision_score(y_true,y_pred,average=average)
            if df['true'].nunique() > 2:
                tmp['auc_ovo'] = roc_auc_score(y_true,y_prab,multi_class='ovo',average=average)
                tmp['auc_ovr'] = roc_auc_score(y_true,y_prab,multi_class='ovr',average=average)
            else:
                tmp['auc'] = roc_auc_score(y_true,y_pred,average=average)
            tmp['f1_score'] = f1_score(y_true,y_pred,average=average)
            tmp['fbeta_score'] = fbeta_score(y_true,y_pred,beta=0.5,average=average)
            
            result.append(tmp)
            
        return pd.DataFrame( result )
    
    def importance(self):
        result = []
        for key in self.m_models:
            model = self.m_models[key]
            df_temp = model.varimp(use_pandas=True)
            if type(df_temp) == pd.DataFrame:
                df_temp['model'] = key
                result.extend(json.loads(df_temp.to_json(orient='records')))
        df_importance = pd.DataFrame(result).reset_index(drop=True)
        return df_importance

def main():

    h2o.init(nthreads = -1, verbose=False)

    df_sample = CFASample.get_random_classification(1000,n_feature=1,n_class=2)
    df_train,df_test = CFASample.split_dataset(df_sample)
    print(df_train)

    model_1 = CFAModelClassify()
    model_1.train(df_train,y_column='y')
    model_1.save("./test")
    model_2 = CFAModelClassify()
    model_2.load("./test")
    df_pred = model_2.predict(df_test,y_column='y')
    print(df_pred)
    df_evaluate = model_2.evaluate(df_test,y_column='y')
    print(df_evaluate)
    df_importance = model_2.importance()
    print(df_importance)
    nan_rows = df_pred[df_pred.isna().any(axis=1)]
    print(nan_rows)
    df_pred = df_pred.dropna()
    print(df_pred)

if __name__ == "__main__":
    main()
