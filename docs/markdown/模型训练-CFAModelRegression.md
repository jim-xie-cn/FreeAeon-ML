# CFAModelRegression - 回归模型训练

## 应用场景

CFAModelRegression类提供自动化的回归模型训练,主要应用于:

- 数值预测(销售额、价格、温度等)
- 趋势分析和预测
- 多模型对比和集成
- 特征重要性分析
- 模型快速原型开发

## 安装依赖

```bash
pip install FreeAeon-ML h2o
```

## 类说明

CFAModelRegression基于H2O框架,自动训练多个回归模型并提供统一接口。

默认包含模型:
- **rf**: 随机森林
- **ann**: 深度神经网络
- **glm**: 广义线性模型
- **gbm**: 梯度提升机
- **xgboost**: XGBoost(部分平台支持)

## 初始化参数

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| ip | str | 'localhost' | H2O服务器IP |
| port | int | 54321 | H2O服务器端口 |
| models | dict | None | 自定义模型字典 |

## 方法详解

### 1. train - 训练模型

```python
def train(self, df_sample, x_columns=[], y_column='y', train_ratio=0.85)
```

训练所有回归模型。

**参数**:
- df_sample: 训练数据
- x_columns: 特征列名列表,空则自动选择
- y_column: 标签列名
- train_ratio: 训练集比例,0则不划分验证集

**示例**:
```python
import h2o
from FreeAeonML.FAModelRegression import CFAModelRegression
from FreeAeonML.FASample import CFASample

h2o.init(nthreads=-1, verbose=False)

df_train = CFASample.get_random_regression(1000)
model = CFAModelRegression()
model.train(df_train, y_column='y', train_ratio=0.85)
```

### 2. predict - 预测

```python
def predict(self, df_sample, x_columns=[], y_column='y')
```

使用所有模型进行预测。

**返回**: DataFrame包含所有模型的预测结果

**示例**:
```python
df_test = CFASample.get_random_regression(200)
df_pred = model.predict(df_test, y_column='y')
print(df_pred.head())
```

### 3. evaluate - 评估模型

```python
def evaluate(self, df_sample, x_columns=[], y_column='y')
```

计算回归评估指标。

**返回**: DataFrame包含以下指标:
- mse: 均方误差
- rmse: 均方根误差
- mae: 平均绝对误差
- r2: R²决定系数

**示例**:
```python
df_eval = model.evaluate(df_test, y_column='y')
print(df_eval)
```

### 4. save/load - 模型持久化

```python
def save(self, model_path)
def load(self, model_path)
```

保存和加载模型。

**示例**:
```python
# 保存
model.save('./regression_models')

# 加载
model2 = CFAModelRegression()
model2.load('./regression_models')
```

### 5. importance - 特征重要性

```python
def importance(self)
```

获取所有模型的特征重要性。

**示例**:
```python
df_importance = model.importance()
print(df_importance)
```

## 完整示例

```python
import h2o
import matplotlib.pyplot as plt
from FreeAeonML.FAModelRegression import CFAModelRegression
from FreeAeonML.FASample import CFASample

# 初始化H2O
h2o.init(nthreads=-1, verbose=False)

# 1. 准备数据
df_sample = CFASample.get_random_regression(1000)
df_train, df_test = CFASample.split_dataset(df_sample)

# 2. 训练模型
model = CFAModelRegression()
model.train(df_train, y_column='y')

# 3. 预测
df_pred = model.predict(df_test, y_column='y')

# 4. 评估
df_eval = model.evaluate(df_test, y_column='y')
print("模型评估:")
print(df_eval)

# 5. 特征重要性
df_imp = model.importance()
print("\n特征重要性:")
print(df_imp)

# 6. 可视化
for model_name in df_pred['model'].unique():
    subset = df_pred[df_pred['model'] == model_name]
    plt.figure(figsize=(8, 6))
    plt.scatter(subset['true'], subset['predict'], alpha=0.5)
    plt.plot([subset['true'].min(), subset['true'].max()],
             [subset['true'].min(), subset['true'].max()], 'r--')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title(f'{model_name} - Prediction vs True')
    plt.show()

# 7. 保存模型
model.save('./my_regression_model')
```

## 注意事项

1. 数据要求: y列必须为数值型
2. 特征选择: 自动排除非数值型列
3. H2O环境: 使用前初始化 h2o.init()
4. 模型选择: 通过evaluate对比选择最佳模型

## 相关类链接

- [CFAModelClassify](./模型训练-CFAModelClassify.md)
- [CFAFeatureSelect](./特征工程-CFAFeatureSelect.md)
