# CFAModelClassify

## 功能分类
模型训练

## 类描述
自动化分类模型训练工具，支持多种分类算法的批量训练、评估和预测

## 应用场景

- 二分类问题：欺诈检测、疾病诊断、客户流失预测
- 多分类问题：文本分类、图像识别、产品分类
- 模型对比：快速对比多种算法的性能
- 自动化建模：减少手动调参和模型选择工作
- 生产部署：训练、保存、加载模型用于预测
        

## 方法列表


### 主要方法

#### 1. __init__(models=None)
初始化分类器，默认包含决策树、随机森林、梯度提升等7种算法。

#### 2. train(df_sample, x_columns=[], y_column='label', train_ratio=0.85)
训练所有模型。

#### 3. predict(df_sample, x_columns=[], y_column='label')
使用训练好的模型进行预测。

#### 4. evaluate(df_sample, x_columns=[], y_column='label', average='weighted')
评估模型性能，返回多个评估指标。

#### 5. importance()
获取特征重要性排序。

#### 6. save(model_path)
保存所有模型到指定路径。

#### 7. load(model_path)
从路径加载所有模型。
        

## 示例代码


import h2o
from FreeAeonML.FAModelClassify import CFAModelClassify
from FreeAeonML.FASample import CFASample

# 初始化H2O
h2o.init(nthreads=-1, verbose=False)

# 1. 准备数据
df_sample = CFASample.get_random_classification(1000, n_feature=5, n_class=2)
df_train, df_test = CFASample.split_dataset(df_sample, train_ratio=0.8)
print(f"训练集大小: {len(df_train)}, 测试集大小: {len(df_test)}")

# 2. 训练模型
model = CFAModelClassify()
model.train(df_train, y_column='y', train_ratio=0.85)

# 3. 保存模型
model.save("./models/classification")

# 4. 加载模型
model_loaded = CFAModelClassify()
model_loaded.load("./models/classification")

# 5. 预测
df_pred = model_loaded.predict(df_test, y_column='y')
print("预测结果:\n", df_pred.head(10))

# 6. 评估模型
df_evaluate = model_loaded.evaluate(df_test, y_column='y', average='weighted')
print("\n模型评估结果:")
print(df_evaluate[['model', 'accuracy', 'precision', 'recall', 'f1_score', 'auc']])

# 7. 特征重要性
df_importance = model_loaded.importance()
print("\n特征重要性:")
for model_name, group in df_importance.groupby('model'):
    print(f"\n{model_name}模型:")
    print(group.nlargest(5, 'scaled_importance')[['variable', 'scaled_importance']])

# 8. 自定义模型集合
from h2o.estimators import H2ORandomForestEstimator, H2OGradientBoostingEstimator
custom_models = {
    "rf": H2ORandomForestEstimator(ntrees=100),
    "gbm": H2OGradientBoostingEstimator(ntrees=50)
}
model_custom = CFAModelClassify(models=custom_models)
model_custom.train(df_train, y_column='y')
        

## 参数说明


| 方法 | 参数 | 类型 | 说明 |
|------|------|------|------|
| __init__ | models | dict | 自定义模型字典，None使用默认 |
| train | df_sample | pd.DataFrame | 训练数据 |
| | x_columns | list | 特征列，默认自动检测 |
| | y_column | str | 目标列名 |
| | train_ratio | float | 训练集比例，0表示全部用于训练 |
| predict | df_sample | pd.DataFrame | 预测数据 |
| | x_columns | list | 特征列 |
| | y_column | str | 真实标签列（可选） |
| evaluate | df_sample | pd.DataFrame | 评估数据 |
| | x_columns | list | 特征列 |
| | y_column | str | 真实标签列 |
| | average | str | 多分类平均方式：weighted/macro/micro |
| save | model_path | str | 模型保存路径 |
| load | model_path | str | 模型加载路径 |
        

## 返回值说明


- **train**: 无返回值
- **predict**: DataFrame包含预测结果和概率
- **evaluate**: DataFrame包含各模型的评估指标
- **importance**: DataFrame包含特征重要性排序
- **save**: 无返回值
- **load**: 无返回值
        

## 注意事项


- 默认支持7种算法：决策树、SVM、随机森林、神经网络、贝叶斯、GLM、梯度提升
- Windows和ARM架构不支持XGBoost
- 评估指标包括：准确率、精确率、召回率、F1分数、AUC、MCC等
- 多分类问题自动计算OVO和OVR的AUC
- 使用前需要初始化H2O环境
- 模型保存为H2O格式，跨平台兼容性好
        

---
*生成时间: 2026-03-23 16:10:12*
*项目: FreeAeon-ML*
