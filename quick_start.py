import numpy as np
import pandas as pd
from FreeAeonML.FADataPreprocess import CFADataPreprocess
from FreeAeonML.FASample import CFASample
from FreeAeonML.FAModelClassify import CFAModelClassify

from h2o.estimators import H2ORandomForestEstimator
import h2o

h2o.init()

# 生成样本数据
df_sample = CFASample.get_random_classification(1000, n_feature=5, n_class=2)
df_train, df_test = CFASample.split_dataset(df_sample)

# 使用系统自带的模型进行训练
model = CFAModelClassify(models=None)

#如果需要使用指定的模型进行训练，请按照以下格式指定模型）
#model = CFAModelClassify(models={"rf": H2ORandomForestEstimator()})

#训练模型
model.train(df_train, y_column="y")

# 模型评估
df_pred = model.predict(df_test, y_column="y")
df_eval = model.evaluate(df_test, y_column="y")
print(df_eval)
