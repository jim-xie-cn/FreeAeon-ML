# CFAModelRegression

## 功能分类
模型训练

## 类描述
自动化回归模型训练工具，支持多种回归算法的批量训练、评估和预测

## 应用场景

- 房价预测：基于房屋特征预测价格
- 销售预测：预测未来销售额
- 股票价格预测：基于历史数据预测走势
- 需求预测：预测产品需求量
- 风险评估：预测连续型风险指标
        

## 方法列表


### 主要方法

#### 1. __init__(models=None)
初始化回归器，默认包含随机森林、神经网络、GLM、梯度提升4种算法。

#### 2. train(df_sample, x_columns=[], y_column='y', train_ratio=0.85)
训练所有模型。

#### 3. predict(df_sample, x_columns=[], y_column='y')
使用训练好的模型进行预测。

#### 4. evaluate(df_sample, x_columns=[], y_column='y')
评估模型性能，返回MSE、RMSE、MAE、R²等指标。

#### 5. importance()
获取特征重要性排序。

#### 6. save(model_path)
保存所有模型到指定路径。

#### 7. load(model_path)
从路径加载所有模型。
        

## 示例代码


import h2o
from FreeAeonML.FAModelRegression import CFAModelRegression
from FreeAeonML.FASample import CFASample

# 初始化H2O
h2o.init(nthreads=-1, verbose=False)

# 1. 准备数据
df_sample = CFASample.get_random_regression(1000)
df_sample['y'] = df_sample['y'].astype(float)
df_train, df_test = CFASample.split_dataset(df_sample, train_ratio=0.8)
print(f"训练集大小: {len(df_train)}, 测试集大小: {len(df_test)}")

# 2. 训练模型
model = CFAModelRegression()
model.train(df_train, y_column='y', train_ratio=0.85)

# 3. 保存模型
model.save("./models/regression")

# 4. 加载模型
model_loaded = CFAModelRegression()
model_loaded.load("./models/regression")

# 5. 预测
df_pred = model_loaded.predict(df_test, y_column='y')
print("预测结果:\n", df_pred.head(10))

# 6. 评估模型
df_evaluate = model_loaded.evaluate(df_test, y_column='y')
print("\n模型评估结果:")
print(df_evaluate[['model', 'mse', 'rmse', 'mae', 'r2']])

# 找出最佳模型
best_model = df_evaluate.loc[df_evaluate['r2'].idxmax()]
print(f"\n最佳模型: {best_model['model']}, R²: {best_model['r2']:.4f}")

# 7. 特征重要性
df_importance = model_loaded.importance()
print("\n特征重要性:")
for model_name, group in df_importance.groupby('model'):
    print(f"\n{model_name}模型:")
    print(group.nlargest(5, 'scaled_importance')[['variable', 'scaled_importance']])

# 8. 自定义模型集合
from h2o.estimators import H2ORandomForestEstimator, H2OGradientBoostingEstimator
custom_models = {
    "rf": H2ORandomForestEstimator(ntrees=200),
    "gbm": H2OGradientBoostingEstimator(ntrees=100, max_depth=8)
}
model_custom = CFAModelRegression(models=custom_models)
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
| | y_column | str | 真实值列（可选） |
| evaluate | df_sample | pd.DataFrame | 评估数据 |
| | x_columns | list | 特征列 |
| | y_column | str | 真实值列 |
| save | model_path | str | 模型保存路径 |
| load | model_path | str | 模型加载路径 |
        

## 返回值说明


- **train**: 无返回值
- **predict**: DataFrame包含预测值和真实值
- **evaluate**: DataFrame包含各模型的评估指标
- **importance**: DataFrame包含特征重要性排序
- **save**: 无返回值
- **load**: 无返回值
        

## 注意事项


- 默认支持4种算法：随机森林、神经网络、GLM、梯度提升
- Windows和ARM架构不支持XGBoost
- 评估指标包括：MSE、RMSE、MAE、R²
- R²越接近1表示模型拟合越好
- MSE/RMSE/MAE越小表示预测误差越小
- 使用前需要初始化H2O环境
- 目标变量必须为数值型（float或int）
        

---
*生成时间: 2026-03-23 16:10:12*
*项目: FreeAeon-ML*
