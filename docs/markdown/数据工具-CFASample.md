# CFASample

## 功能分类
数据工具

## 类描述
样本数据生成和处理工具类，提供生成测试数据、数据集划分、数据重采样等功能

## 应用场景

- 机器学习模型训练前需要生成测试数据
- 需要将数据集划分为训练集和测试集
- 处理类别不平衡问题需要数据重采样
- 时间序列数据需要找到变化点
- 大数据集需要分批处理
        

## 方法列表


### get_random_regression(n_sample=100)
生成回归问题的随机样本数据。

### get_random_classification(n_sample=100, n_feature=2, n_class=2)
生成分类问题的随机样本数据。

### get_random_cluster(n_sample=100)
生成聚类问题的随机样本数据。

### split_dataset(df, test_ratio=0.2)
将数据集划分为训练集和测试集。

### resample_smote(df_sample, x_columns=[], y_column='label', random_state=42)
使用SMOTE算法对少数类进行过采样。

### resample_balance(df_data, y_column='labels')
通过上采样平衡数据集中的类别分布。

### find_changed_index(ds)
找到序列中发生变化的索引位置。

### split_dataframe(df_sample, batch_size=10)
将DataFrame划分成多个批次。
        

## 示例代码


from FreeAeonML.FASample import CFASample

# 1. 生成分类样本数据
df = CFASample.get_random_classification(1000, n_feature=10, n_class=2)

# 2. 划分数据集
df_train, df_test = CFASample.split_dataset(df, test_ratio=0.2)

# 3. SMOTE重采样
df_balanced = CFASample.resample_smote(df, y_column='y')

# 4. 批量处理
batches = CFASample.split_dataframe(df_train, batch_size=100)
        

## 参数说明


| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| n_sample | int | 100 | 样本数量 |
| n_feature | int | 2 | 特征数量 |
| n_class | int | 2 | 类别数量 |
| test_ratio | float | 0.2 | 测试集比例 |
        

## 返回值说明

返回 DataFrame 或 tuple(DataFrame, DataFrame)

## 注意事项


- SMOTE重采样适用于类别不平衡的分类问题
- split_dataset 会自动重置索引
        

---
*生成时间: 2026-03-23 16:10:12*
*项目: FreeAeon-ML*
