# CFAVisualize - 数据可视化

## 应用场景

本模块包含两个类:
- **CFAVisualize**: 常用可视化方法
- **CFANetGraph**: 网络图和序列可视化

主要应用于:
- 热力图展示
- 桑基图(流向图)
- 等高线图
- 时间序列可视化
- 关系网络图

## 安装依赖

```bash
pip install FreeAeon-ML pyecharts pyvis
```

## CFAVisualize类

### 1. show_heatmap - 热力图

```python
@staticmethod
def show_heatmap(df, title="heatmap demo")
```

绘制热力图,适合展示相关性矩阵或数据表格。

**示例**:
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from FreeAeonML.FAVisualize import CFAVisualize

# 创建相关性数据
df = pd.DataFrame(np.random.rand(10, 10))
CFAVisualize.show_heatmap(df, title="Correlation Matrix")
plt.show()
```

### 2. get_sankey - 桑基图

```python
@staticmethod
def get_sankey(df_data_list, title="sankey demo")
```

生成桑基图用于展示流向关系。

**参数**:
- df_data_list: DataFrame或DataFrame列表,包含源、目标、流量三列

**示例**:
```python
import pandas as pd
from FreeAeonML.FAVisualize import CFAVisualize
import webbrowser
import os

# 准备数据
data = pd.DataFrame({
    '来源': ['A', 'A', 'B', 'B', 'C'],
    '目标': ['X', 'Y', 'X', 'Y', 'Y'],
    '流量': [100, 50, 80, 30, 60]
})

# 生成桑基图
sankey = CFAVisualize.get_sankey(data, title="流向分析")

# 保存并打开
file_path = os.path.abspath("sankey.html")
sankey.render(file_path)
webbrowser.open(f"file://{file_path}")
```

### 3. show_contour - 等高线图

```python
@staticmethod
def show_contour(df_data, isBlack=True, withTips=True)
```

绘制等高线图,适合展示三维数据。

**参数**:
- df_data: 包含(x, y, value)三列的DataFrame
- isBlack: 是否黑白显示
- withTips: 是否显示数值标签

**示例**:
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from FreeAeonML.FAVisualize import CFAVisualize

# 生成网格数据
data_list = []
for i in range(20):
    for j in range(20):
        value = np.sin(i/3) * np.cos(j/3)
        data_list.append({'x': i, 'y': j, 'value': value})

df = pd.DataFrame(data_list)

# 绘制等高线
CFAVisualize.show_contour(df, isBlack=False, withTips=False)
plt.colorbar()
plt.title('Contour Plot')
plt.show()
```

### 4. show_sequence - 序列可视化

```python
@staticmethod
def show_sequence(df_data, call_edge_info=None, file_name='pyvis.html')
```

将时间序列可视化为网络图。

**参数**:
- df_data: 必须包含'label','value','title'列
- call_edge_info: 自定义边信息的回调函数
- file_name: 输出HTML文件名

**示例**:
```python
import pandas as pd
from FreeAeonML.FAVisualize import CFAVisualize

# 创建序列数据
data = []
for i in range(10):
    data.append({
        'label': f'步骤{i}',
        'value': i * 10,
        'title': f'这是第{i}个节点'
    })

df = pd.DataFrame(data)

# 可视化
CFAVisualize.show_sequence(df, file_name='my_sequence.html')
```

## CFANetGraph类

### 初始化

```python
net = CFANetGraph(directed=True)
```

**参数**:
- directed: True为有向图,False为无向图

### 方法说明

#### 1. Add_Node - 添加节点

```python
def Add_Node(self, label, value, title, group=None, size=None, color='#00ff1e', node_id=None)
```

#### 2. Add_Edge - 添加边

```python
def Add_Edge(self, from_id, to_id, weight, label)
```

#### 3. Show - 显示网络图

```python
def Show(self, file_name='pyvis.html')
```

### 完整示例

```python
from FreeAeonML.FAVisualize import CFANetGraph

# 创建网络图
net = CFANetGraph(directed=True)

# 添加节点
net.Add_Node(label="开始", value=10, title="起始节点", color='#ff0000')
net.Add_Node(label="处理", value=20, title="处理节点", color='#00ff00')
net.Add_Node(label="结束", value=15, title="结束节点", color='#0000ff')

# 添加边
net.Add_Edge(from_id=1, to_id=2, weight=5, label="流程1")
net.Add_Edge(from_id=2, to_id=3, weight=8, label="流程2")

# 显示
net.Show('network.html')
```

## 综合示例

### 示例1: 数据分析仪表板

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from FreeAeonML.FAVisualize import CFAVisualize

# 生成示例数据
np.random.seed(42)
n = 100

# 热力图:相关性矩阵
corr_data = np.random.rand(5, 5)
corr_data = (corr_data + corr_data.T) / 2
np.fill_diagonal(corr_data, 1)
df_corr = pd.DataFrame(corr_data,
                       columns=['特征A', '特征B', '特征C', '特征D', '特征E'],
                       index=['特征A', '特征B', '特征C', '特征D', '特征E'])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 热力图
plt.sca(axes[0])
CFAVisualize.show_heatmap(df_corr, title="特征相关性")

# 等高线图
data_list = []
for i in np.linspace(-5, 5, 50):
    for j in np.linspace(-5, 5, 50):
        value = np.exp(-(i**2 + j**2)/10) * np.cos(i) * np.sin(j)
        data_list.append({'x': i, 'y': j, 'value': value})

df_contour = pd.DataFrame(data_list)
plt.sca(axes[1])
CFAVisualize.show_contour(df_contour, isBlack=False, withTips=False)

plt.tight_layout()
plt.show()
```

### 示例2: 业务流程分析

```python
import pandas as pd
from FreeAeonML.FAVisualize import CFAVisualize
import webbrowser
import os

# 多级流向数据
df = pd.DataFrame({
    '部门': ['销售', '销售', '销售', '生产', '生产', '物流'],
    '产品': ['产品A', '产品B', '产品C', '产品A', '产品B', '产品A'],
    '区域': ['华东', '华东', '华北', '华东', '华北', '华东'],
    '数量': [1000, 800, 600, 1000, 800, 1000]
})

# 生成桑基图
sankey = CFAVisualize.get_sankey(df, title="业务流向分析")
file_path = os.path.abspath("business_flow.html")
sankey.render(file_path)
webbrowser.open(f"file://{file_path}")
```

## 注意事项

1. **热力图**: 数据应为数值型矩阵
2. **桑基图**: 节点名称不能重复,否则会显示异常
3. **等高线图**: 需要(x,y,value)三列数据
4. **序列图**: 必须包含'label','value','title'列
5. **网络图**: 生成HTML文件,需要浏览器打开

## 相关类链接

- [CFADataTest](./数据探索-CFADataTest.md)
- [CFAFeatureSelect](./特征工程-CFAFeatureSelect.md)
