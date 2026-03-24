# CFAFeatureSelect - 特征选择

## 应用场景

CFAFeatureSelect类提供多种特征选择和降维方法,主要应用于:

- 分类问题的特征重要性分析(Infogram)
- 回归问题的方差分析(ANOVA-GLM)
- 时间序列因果关系检验(Granger Test)
- 高维数据降维(PCA, t-SNE)
- 特征空间可视化

## 安装依赖

```bash
pip install FreeAeon-ML h2o
```

或手动安装依赖包:

```bash
pip install numpy pandas h2o statsmodels scikit-learn matplotlib seaborn
```

## 类说明

CFAFeatureSelect基于H2O框架,提供自动化的特征分析功能。

## 初始化参数

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| ip | str | 'localhost' | H2O服务器IP |
| port | int | 54321 | H2O服务器端口 |

## 方法详解

### 1. load - 加载数据

将pandas DataFrame转换为H2O Frame并指定特征和标签。

#### 参数说明

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| df_sample | pd.DataFrame | - | 样本数据 |
| x_columns | list | [] | 特征列名,空则自动选择 |
| y_column | str | 'label' | 标签列名 |
| is_regression | bool | False | True为回归,False为分类 |

#### 示例代码

```python
import h2o
from FreeAeonML.FAFeatureSelect import CFAFeatureSelect
from FreeAeonML.FASample import CFASample

# 初始化H2O
h2o.init(nthreads=-1, verbose=False)

# 加载分类数据
df_sample = CFASample.get_random_classification(1000, n_feature=10, n_class=2)
fs = CFAFeatureSelect()
fs.load(df_sample, y_column='y', is_regression=False)
```

### 2. get_inform_graph - 信息图分析

生成Infogram用于特征重要性和独特性分析(仅用于分类问题)。

#### 参数说明

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| algorithm | str | 'AUTO' | 算法:'AUTO','deeplearning','drf','gbm','glm','xgboost','All' |
| protected_columns | list | [] | 受保护列(不被移除) |

#### 返回值

- algorithm='All': 返回dict,键为算法名,值为Infogram对象
- 其他: 返回单个Infogram对象

#### Infogram解读

- **横轴(Total Information)**: 特征对预测的总影响力
- **纵轴(Net Information)**: 特征的独特性(去除冗余后的贡献)
- **可接受特征**: 位于虚线右上方的特征,既重要又独特

#### 示例代码

```python
import h2o
from FreeAeonML.FAFeatureSelect import CFAFeatureSelect
from FreeAeonML.FASample import CFASample

h2o.init(nthreads=-1, verbose=False)

# 生成数据
df_sample = CFASample.get_random_classification(1000, n_feature=10, n_class=2)

# 特征选择
fs = CFAFeatureSelect()
fs.load(df_sample, y_column='y', is_regression=False)

# 单算法Infogram
ig = fs.get_inform_graph(algorithm='AUTO')
ig.plot()

# 获取可接受特征
admissible_features = ig.get_admissible_features()
print("推荐特征:", admissible_features)

# 获取特征评分
score_frame = ig.get_admissible_score_frame()
print(score_frame)
```

### 3. get_anovaglm - ANOVA方差分析

统计自变量和因变量的相关性(用于回归问题)。

#### 参数说明

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| family | str | 'gaussian' | 分布族:'gaussian','binomial','poisson'等 |
| lambda_ | float | 0 | 正则化参数 |
| highest_interaction_term | int | 2 | 最高交互项阶数 |

#### 返回值

返回H2OANOVAGLMEstimator模型对象。

#### 示例代码

```python
import h2o
from FreeAeonML.FAFeatureSelect import CFAFeatureSelect
from FreeAeonML.FASample import CFASample

h2o.init(nthreads=-1, verbose=False)

# 回归数据
df_sample = CFASample.get_random_regression(1000)

fs = CFAFeatureSelect()
fs.load(df_sample, y_column='y', is_regression=True)

# ANOVA分析
anova_model = fs.get_anovaglm(
    family='gaussian',
    lambda_=0,
    highest_interaction_term=2
)

# 查看结果
print(anova_model.summary())
```

### 4. granger_test - 格兰杰因果检验(静态方法)

检验时间序列的因果关系。

#### 参数说明

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| ds_result | pd.Series | - | 结果序列(必须平稳) |
| ds_source | pd.Series | - | 原因序列(必须平稳) |
| maxlag | int/list | - | 最大滞后阶数 |
| p_value | float | 0.05 | 显著性水平 |

#### 返回值

返回元组(最小p值, 最佳lag, 通过列表, 详细结果)。

#### 示例代码

```python
import numpy as np
import pandas as pd
from FreeAeonML.FAFeatureSelect import CFAFeatureSelect

# 生成因果关系数据
np.random.seed(42)
n = 100
source = np.random.randn(n)
result = np.zeros(n)
for i in range(2, n):
    result[i] = 0.5 * source[i-1] + 0.3 * source[i-2] + np.random.randn() * 0.1

# 格兰杰检验
min_p, best_lag, approve_list, detail = CFAFeatureSelect.granger_test(
    ds_result=pd.Series(result),
    ds_source=pd.Series(source),
    maxlag=5,
    p_value=0.05
)

print(f"最小p值: {min_p:.6f}")
print(f"最佳滞后: {best_lag}")
print(f"通过检验的lag: {approve_list}")
```

### 5. get_data_pca - PCA和t-SNE降维(静态方法)

对高维数据进行降维和可视化。

#### 参数说明

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| df_samples | pd.DataFrame | - | 样本数据 |
| n_components | int | 2 | 降维目标维度(2或3) |
| label_column | str | 'y' | 标签列名 |
| feature_list | list | [] | 特征列,空则自动选择 |
| with_plot | bool | True | 是否绘图 |
| perplexity | int | None | t-SNE perplexity参数 |
| n_clusters | int | 2 | 无标签时的聚类数 |

#### 返回值

返回元组(PCA降维DataFrame, t-SNE降维DataFrame)。

#### 示例代码

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from FreeAeonML.FAFeatureSelect import CFAFeatureSelect

# 生成高维数据
np.random.seed(42)
n_samples = 500
n_features = 50

data = np.random.rand(n_samples, n_features)
labels = np.random.randint(0, 3, n_samples)

df = pd.DataFrame(data, columns=[f'f{i}' for i in range(n_features)])
df['y'] = labels

# 降维可视化(2D)
df_pca, df_tsne = CFAFeatureSelect.get_data_pca(
    df,
    n_components=2,
    label_column='y',
    with_plot=True
)

# 降维可视化(3D)
df_pca_3d, df_tsne_3d = CFAFeatureSelect.get_data_pca(
    df,
    n_components=3,
    label_column='y',
    with_plot=True
)

plt.show()

print("PCA降维结果:")
print(df_pca.head())
print("\nt-SNE降维结果:")
print(df_tsne.head())
```

### 6. get_matrix_by_pca - PCA矩阵分解(静态方法)

使用PCA进行矩阵分解和重构。

#### 参数说明

| 参数名 | 类型 | 说明 |
|--------|------|------|
| dataMat | np.ndarray | 原始数据矩阵 |
| n | int | 保留的主成分数 |

#### 返回值

返回元组(降维后矩阵, 重构矩阵)。

#### 示例代码

```python
import numpy as np
from FreeAeonML.FAFeatureSelect import CFAFeatureSelect

# 生成数据
np.random.seed(42)
data = np.random.rand(100, 10)

# PCA降维到2维
lowD_data, recon_data = CFAFeatureSelect.get_matrix_by_pca(data, n=2)

print("原始数据形状:", data.shape)
print("降维后形状:", lowD_data.shape)
print("重构后形状:", recon_data.shape)

# 计算重构误差
recon_error = np.mean((data - recon_data) ** 2)
print(f"重构误差(MSE): {recon_error:.6f}")
```

## 完整示例

### 示例1: 分类问题特征选择流程

```python
import h2o
import matplotlib.pyplot as plt
from FreeAeonML.FAFeatureSelect import CFAFeatureSelect
from FreeAeonML.FASample import CFASample

# 初始化H2O
h2o.init(nthreads=-1, verbose=False)

# 生成数据(10个特征,部分无用)
df_sample = CFASample.get_random_classification(1000, n_feature=10, n_class=2)

# 特征选择分析
fs = CFAFeatureSelect()
fs.load(df_sample, y_column='y', is_regression=False)

# 方法1: 单算法Infogram
print("=" * 60)
print("方法1: AUTO算法Infogram")
ig_auto = fs.get_inform_graph(algorithm='AUTO')
ig_auto.plot()

admissible = ig_auto.get_admissible_features()
print(f"推荐特征: {admissible}")

# 方法2: 多算法对比
print("\n" + "=" * 60)
print("方法2: 多算法对比")
ig_dict = fs.get_inform_graph(algorithm='All')

for algo, ig_obj in ig_dict.items():
    features = ig_obj.get_admissible_features()
    print(f"{algo}: {features}")

plt.show()
```

### 示例2: 回归问题方差分析

```python
import h2o
from FreeAeonML.FAFeatureSelect import CFAFeatureSelect
from FreeAeonML.FASample import CFASample

h2o.init(nthreads=-1, verbose=False)

# 生成回归数据
df_sample = CFASample.get_random_regression(1000)

# ANOVA分析
fs = CFAFeatureSelect()
fs.load(df_sample, y_column='y', is_regression=True)

anova_model = fs.get_anovaglm(
    family='gaussian',
    lambda_=0,
    highest_interaction_term=2
)

# 查看显著性检验
print(anova_model.summary())

# 提取p值小于0.05的特征
result = anova_model.result()
significant_features = []
for key in result.keys():
    if 'p-value' in key and result[key] < 0.05:
        significant_features.append(key)

print(f"\n显著特征(p<0.05): {significant_features}")
```

### 示例3: 高维数据可视化

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from FreeAeonML.FAFeatureSelect import CFAFeatureSelect
from sklearn.datasets import load_digits

# 加载手写数字数据(64维)
digits = load_digits()
df = pd.DataFrame(digits.data)
df['label'] = digits.target

print(f"原始数据维度: {digits.data.shape}")

# PCA和t-SNE降维可视化
df_pca, df_tsne = CFAFeatureSelect.get_data_pca(
    df,
    n_components=2,
    label_column='label',
    with_plot=True,
    perplexity=30
)

plt.show()

# 保存降维结果用于后续建模
print("\nPCA降维后数据:")
print(df_pca.head())
```

### 示例4: 时间序列因果分析

```python
import numpy as np
import pandas as pd
from FreeAeonML.FAFeatureSelect import CFAFeatureSelect

# 生成多个时间序列
np.random.seed(42)
n = 200

# 基础序列
base = np.random.randn(n)

# 派生序列(有因果关系)
series_dict = {}
series_dict['base'] = base

series_dict['lag1'] = np.zeros(n)
for i in range(1, n):
    series_dict['lag1'][i] = 0.7 * base[i-1] + np.random.randn() * 0.3

series_dict['lag2'] = np.zeros(n)
for i in range(2, n):
    series_dict['lag2'][i] = 0.5 * base[i-2] + np.random.randn() * 0.3

# 无关序列
series_dict['random'] = np.random.randn(n)

# 因果检验
print("格兰杰因果检验:")
print("=" * 60)

for name, series in series_dict.items():
    if name == 'base':
        continue

    min_p, best_lag, approve_list, detail = CFAFeatureSelect.granger_test(
        ds_result=pd.Series(series),
        ds_source=pd.Series(base),
        maxlag=5,
        p_value=0.05
    )

    print(f"\n{name}序列:")
    print(f"  最小p值: {min_p:.6f}")
    print(f"  最佳滞后: {best_lag}")
    print(f"  因果关系: {'存在' if min_p < 0.05 else '不存在'}")
```

## 注意事项

1. **H2O环境**:
   - 使用前需要初始化: `h2o.init()`
   - WSL环境使用: `h2o.connect(ip='localhost', port=54321)`
   - 使用完毕调用: `h2o.shutdown()`

2. **Infogram分析**:
   - 仅适用于分类问题
   - 需要足够样本(建议>100)
   - protected_columns可保护重要业务特征

3. **ANOVA分析**:
   - 仅适用于回归问题
   - p值<0.05表示显著相关
   - highest_interaction_term不宜过大

4. **格兰杰检验**:
   - 要求序列平稳
   - maxlag不宜过大(建议<数据长度/10)
   - 统计意义的因果≠真实因果

5. **PCA降维**:
   - 适合线性关系数据
   - t-SNE适合非线性,但计算慢
   - perplexity影响t-SNE聚类效果

## 相关类链接

- [CFAModelClassify](./模型训练-CFAModelClassify.md) - 分类模型训练
- [CFAModelRegression](./模型训练-CFAModelRegression.md) - 回归模型训练
- [CFADataTest](./数据探索-CFADataTest.md) - 数据假设检验(包含格兰杰检验)
