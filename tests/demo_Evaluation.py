import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json,os,sys
import webbrowser
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from FreeAeonML.FAEvaluation import CFAEvaluation
from FreeAeonML.FASample import CFASample

def main():

    df_sample = CFASample.get_random_classification(1000, n_feature=10, n_class=2)
    df_train, df_test = CFASample.split_dataset(df_sample)

    X = df_train.drop(columns=['y'])
    y = df_train['y']
    X_test = df_test.drop(columns=['y'])
    y_test = df_test['y']

    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()
    model.fit(X, y)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # 评估结果
    eval_result = CFAEvaluation.evaluate(pd.Series(y_pred), pd.Series(y_test))
    print(eval_result)

    # 获取并绘制 ROC 曲线
    roc_auc, df_roc = CFAEvaluation.get_binary_roc(y_test, y_prob)
    CFAEvaluation.show_binary_roc(roc_auc, df_roc, title="Logistic Regression ROC")

if __name__ == "__main__":
    main()
