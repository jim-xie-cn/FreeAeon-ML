# CFANetGraph - 网络图可视化类

## 应用场景

CFANetGraph类专门用于创建和可视化网络图，适用于：

- 时间序列数据的节点流向可视化
- 业务流程图和工作流展示
- 关系网络分析和展示
- 状态转移图可视化
- 有向图和无向图的创建
- 交互式网络图展示（基于pyvis）

## 安装依赖

```bash
pip install pyvis networkx pandas numpy
```

## 类说明

### CFANetGraph

网络图构建和可视化类，支持节点和边的添加、属性设置，并可导出为交互式HTML。

**支持特性**：
- 有向图和无向图
- 节点自定义（颜色、大小、标签、提示信息）
- 边自定义（权重、标签）
- 交互式HTML输出
- 自动浏览器打开

## 初始化

### `__init__(directed=True)`

**参数**：
- `directed` (bool, 默认=True): True为有向图，False为无向图

**示例**：
```python
from FreeAeonML.FAVisualize import CFANetGraph

# 创建有向图
net = CFANetGraph(directed=True)

# 创建无向图
net_undirected = CFANetGraph(directed=False)
```

## 方法详解

### 1. Add_Node - 添加节点

**功能**：向图中添加一个节点。

**调用参数**：
- `label` (str): 节点显示标签
- `value` (float/int): 节点值（影响节点大小）
- `title` (str): 悬停时显示的提示信息
- `group` (str, 默认=None): 节点分组（相同组的节点颜色相同）
- `size` (int, 默认=None): 节点大小
- `color` (str, 默认='#00ff1e'): 节点颜色（十六进制）
- `node_id` (int, 默认=None): 节点ID，不指定则自动生成

**返回值**：
- int: 节点ID

**示例代码**：
```python
from FreeAeonML.FAVisualize import CFANetGraph

net = CFANetGraph()

# 添加基本节点
node1_id = net.Add_Node(
    label="起点",
    value=100,
    title="这是起点节点"
)

# 添加自定义样式节点
node2_id = net.Add_Node(
    label="终点",
    value=200,
    title="这是终点节点",
    color='#ff0000',  # 红色
    group='endpoint'
)

print(f"节点1 ID: {node1_id}, 节点2 ID: {node2_id}")
```

### 2. Add_Edge - 添加边

**功能**：在两个节点之间添加一条边。

**调用参数**：
- `from_id` (int): 起始节点ID
- `to_id` (int): 目标节点ID
- `weight` (float): 边的权重（影响边的粗细）
- `label` (str): 边的标签

**返回值**：无

**示例代码**：
```python
from FreeAeonML.FAVisualize import CFANetGraph

net = CFANetGraph()

# 添加节点
n1 = net.Add_Node("A", 100, "节点A")
n2 = net.Add_Node("B", 200, "节点B")
n3 = net.Add_Node("C", 150, "节点C")

# 添加边
net.Add_Edge(n1, n2, weight=5, label="A到B")
net.Add_Edge(n2, n3, weight=3, label="B到C")
net.Add_Edge(n1, n3, weight=2, label="A到C")
```

### 3. Add - 便捷添加方法

**功能**：同时添加两个节点及其之间的边（如果节点已存在则不重复添加）。

**调用参数**：
- `from_node` (dict): 起始节点属性字典
- `to_node` (dict): 目标节点属性字典
- `title` (str, 默认=None): 边的标签
- `weight` (float, 默认=None): 边的权重

**节点字典属性**：
- `id`: 节点ID（可选）
- `label`: 显示标签
- `value`: 节点值
- `title`: 提示信息
- `group`: 分组
- `size`: 大小
- `color`: 颜色

**返回值**：
- tuple: (from_id, to_id)

**示例代码**：
```python
from FreeAeonML.FAVisualize import CFANetGraph

net = CFANetGraph()

# 定义节点
node_a = {
    'label': '步骤A',
    'value': 100,
    'title': '第一步',
    'color': '#3498db'
}

node_b = {
    'label': '步骤B',
    'value': 150,
    'title': '第二步',
    'color': '#e74c3c'
}

# 添加节点和边
id_a, id_b = net.Add(node_a, node_b, title="流向", weight=10)

# 继续添加
node_c = {'label': '步骤C', 'value': 200, 'title': '第三步'}
net.Add(node_b, node_c, title="下一步", weight=5)
```

### 4. Show - 显示图形

**功能**：生成交互式HTML并保存到文件。

**调用参数**：
- `file_name` (str, 默认='pyvis.html'): 输出HTML文件名

**返回值**：无

**示例代码**：
```python
from FreeAeonML.FAVisualize import CFANetGraph

net = CFANetGraph()

# 构建图
n1 = net.Add_Node("A", 100, "节点A")
n2 = net.Add_Node("B", 200, "节点B")
net.Add_Edge(n1, n2, weight=5, label="连接")

# 生成HTML
net.Show('my_network.html')
# 文件会保存在当前目录
```

### 5. Show_Series - 静态方法：时序数据可视化

**功能**：将DataFrame格式的时序数据可视化为网络图，自动按行顺序连接节点。

**调用参数**：
- `df_in` (DataFrame): 时序数据，必须包含列：
  - `label`: 节点标签
  - `value`: 节点值
  - `title`: 提示信息
  - `group` (可选): 分组
  - `size` (可选): 大小
  - `color` (可选): 颜色
- `call_edge_info` (function, 默认=None): 自定义边信息的回调函数
- `file_name` (str, 默认='pyvis.html'): 输出文件名

**回调函数签名**：
```python
def custom_edge_info(from_node, to_node):
    """
    计算边的权重和标签

    Args:
        from_node: 起始节点字典
        to_node: 目标节点字典

    Returns:
        tuple: (weight, title)
    """
    weight = max(from_node['value'], to_node['value'])
    title = f"{from_node['value']}→{to_node['value']}"
    return weight, title
```

**示例代码**：
```python
from FreeAeonML.FAVisualize import CFANetGraph
import pandas as pd

# 创建时序数据
data = []
for i in range(10):
    data.append({
        'label': f'时刻{i}',
        'value': i * 10,
        'title': f'第{i}个时刻\n值={i*10}',
        'color': '#3498db' if i % 2 == 0 else '#e74c3c'
    })

df_data = pd.DataFrame(data)

# 自定义边信息
def my_edge_info(from_node, to_node):
    diff = to_node['value'] - from_node['value']
    weight = abs(diff)
    title = f"变化: +{diff}"
    return weight, title

# 生成时序网络图
CFANetGraph.Show_Series(
    df_data,
    call_edge_info=my_edge_info,
    file_name='timeline.html'
)
```

### 6. call_edge_info - 静态方法：默认边信息计算

**功能**：默认的边信息计算方法（Show_Series使用）。

**调用参数**：
- `from_node` (dict): 起始节点
- `to_node` (dict): 目标节点

**返回值**：
- tuple: (weight, title)

**示例代码**：
```python
from FreeAeonML.FAVisualize import CFANetGraph

# 测试默认边信息计算
node1 = {'value': 100}
node2 = {'value': 200}

weight, title = CFANetGraph.call_edge_info(node1, node2)
print(f"权重: {weight}, 标签: {title}")
# 输出: 权重: 200, 标签: 100:200
```

## 完整示例

### 示例1：简单的流程图

```python
from FreeAeonML.FAVisualize import CFANetGraph

# 创建有向图
net = CFANetGraph(directed=True)

# 添加节点
start = net.Add_Node("开始", 100, "流程起点", color='#2ecc71')
process1 = net.Add_Node("数据处理", 150, "清洗数据", color='#3498db')
process2 = net.Add_Node("特征工程", 200, "提取特征", color='#3498db')
decision = net.Add_Node("模型训练", 250, "训练模型", color='#f39c12')
end = net.Add_Node("结束", 100, "流程终点", color='#e74c3c')

# 添加边
net.Add_Edge(start, process1, weight=5, label="开始")
net.Add_Edge(process1, process2, weight=8, label="处理完成")
net.Add_Edge(process2, decision, weight=10, label="特征就绪")
net.Add_Edge(decision, end, weight=5, label="训练完成")

# 生成HTML
net.Show('workflow.html')
print("流程图已生成: workflow.html")
```

### 示例2：分组网络图

```python
from FreeAeonML.FAVisualize import CFANetGraph

net = CFANetGraph(directed=False)

# 第一组节点（蓝色）
for i in range(1, 4):
    net.Add_Node(
        f"A{i}",
        value=100 + i*10,
        title=f"A组节点{i}",
        group='group_a',
        color='#3498db'
    )

# 第二组节点（红色）
for i in range(1, 4):
    net.Add_Node(
        f"B{i}",
        value=100 + i*15,
        title=f"B组节点{i}",
        group='group_b',
        color='#e74c3c'
    )

# 组内连接
net.Add_Edge(1, 2, weight=5, label="A1-A2")
net.Add_Edge(2, 3, weight=5, label="A2-A3")
net.Add_Edge(4, 5, weight=5, label="B1-B2")
net.Add_Edge(5, 6, weight=5, label="B2-B3")

# 组间连接
net.Add_Edge(1, 4, weight=8, label="A1-B1")
net.Add_Edge(3, 6, weight=8, label="A3-B3")

net.Show('group_network.html')
```

### 示例3：时间序列可视化

```python
from FreeAeonML.FAVisualize import CFANetGraph
import pandas as pd
import numpy as np

# 生成模拟股价数据
np.random.seed(42)
dates = pd.date_range('2024-01-01', periods=30, freq='D')
prices = 100 + np.cumsum(np.random.randn(30) * 2)

# 创建数据框
data = []
for date, price in zip(dates, prices):
    trend = "上涨" if price > 100 else "下跌"
    data.append({
        'label': date.strftime('%m-%d'),
        'value': abs(price),
        'title': f"日期: {date.strftime('%Y-%m-%d')}\n价格: {price:.2f}\n趋势: {trend}",
        'color': '#2ecc71' if price > 100 else '#e74c3c'
    })

df_price = pd.DataFrame(data)

# 自定义边的计算方式
def price_edge_info(from_node, to_node):
    from_val = from_node['value']
    to_val = to_node['value']
    change = to_val - from_val
    weight = abs(change)

    if change > 0:
        title = f"涨 {change:.2f}"
    else:
        title = f"跌 {abs(change):.2f}"

    return weight, title

# 生成时序网络图
CFANetGraph.Show_Series(
    df_price,
    call_edge_info=price_edge_info,
    file_name='stock_timeline.html'
)

print("股价时序图已生成: stock_timeline.html")
```

### 示例4：使用Add方法快速构建

```python
from FreeAeonML.FAVisualize import CFANetGraph

net = CFANetGraph()

# 定义流程节点
steps = [
    {'id': 1, 'label': '需求分析', 'value': 100, 'title': '分析业务需求'},
    {'id': 2, 'label': '系统设计', 'value': 150, 'title': '架构设计'},
    {'id': 3, 'label': '编码开发', 'value': 200, 'title': '代码实现'},
    {'id': 4, 'label': '测试验证', 'value': 150, 'title': '功能测试'},
    {'id': 5, 'label': '上线部署', 'value': 100, 'title': '生产环境部署'}
]

# 快速连接
for i in range(len(steps) - 1):
    net.Add(
        steps[i],
        steps[i+1],
        title=f"步骤{i+1}→{i+2}",
        weight=10
    )

net.Show('development_process.html')
```

### 示例5：复杂关系网络

```python
from FreeAeonML.FAVisualize import CFANetGraph

net = CFANetGraph(directed=True)

# 中心节点
center = net.Add_Node(
    "核心系统",
    value=500,
    title="中央处理系统",
    color='#e74c3c',
    size=50
)

# 添加子系统
subsystems = []
colors = ['#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
for i, (name, color) in enumerate(zip(['用户', '订单', '支付', '库存', '物流'], colors)):
    node_id = net.Add_Node(
        name + "系统",
        value=200,
        title=f"{name}子系统",
        color=color,
        group=name
    )
    subsystems.append(node_id)

    # 连接到中心
    net.Add_Edge(center, node_id, weight=8, label=f"管理{name}")
    net.Add_Edge(node_id, center, weight=5, label=f"上报{name}数据")

# 子系统间的关联
net.Add_Edge(subsystems[0], subsystems[1], weight=6, label="下单")
net.Add_Edge(subsystems[1], subsystems[2], weight=6, label="付款")
net.Add_Edge(subsystems[1], subsystems[3], weight=6, label="扣库存")
net.Add_Edge(subsystems[3], subsystems[4], weight=6, label="发货")

net.Show('system_architecture.html')
```

## 可视化效果说明

生成的HTML文件具有以下交互特性：

- **拖拽节点**：鼠标拖动节点可以重新布局
- **缩放**：鼠标滚轮缩放视图
- **悬停提示**：鼠标悬停在节点上显示title信息
- **节点大小**：根据value值自动调整
- **边粗细**：根据weight值自动调整
- **颜色分组**：相同group的节点自动聚类

## HTML输出示例

```html
<!-- 生成的HTML文件会包含交互式网络图 -->
<!-- 可以在任何现代浏览器中打开 -->
<!-- 支持拖拽、缩放、悬停等交互 -->
```

## 颜色参考

常用的十六进制颜色代码：

| 颜色 | 代码 | 说明 |
|------|------|------|
| 绿色 | #2ecc71 | 成功、正常 |
| 蓝色 | #3498db | 信息、处理中 |
| 红色 | #e74c3c | 错误、警告 |
| 橙色 | #f39c12 | 待处理 |
| 紫色 | #9b59b6 | 特殊状态 |
| 青色 | #1abc9c | 辅助信息 |
| 灰色 | #95a5a6 | 禁用、无效 |

## 注意事项

1. **节点ID唯一性**：相同ID的节点只会保留第一个
2. **边的方向**：有向图中边有方向，无向图中边无方向
3. **文件路径**：Show方法生成的HTML保存在当前工作目录
4. **浏览器打开**：Show_Series会自动在浏览器中打开生成的HTML
5. **数据格式**：Show_Series要求DataFrame必须包含label、value、title列
6. **性能考虑**：节点数超过1000时可能影响浏览器性能
7. **中文显示**：确保HTML文件以UTF-8编码保存

## 典型应用场景

### 场景1：业务流程可视化
```python
# 订单处理流程
# 起点 → 下单 → 支付 → 发货 → 签收 → 完成
```

### 场景2：系统架构图
```python
# 微服务架构
# API网关 → 用户服务/订单服务/支付服务 → 数据库
```

### 场景3：时序数据追踪
```python
# 追踪某个指标的变化趋势
# 时刻1 → 时刻2 → 时刻3 → ...
```

### 场景4：依赖关系图
```python
# 模块依赖关系
# ModuleA → ModuleB → ModuleC
```

## 相关类

- [CFAVisualize](可视化-CFAVisualize.md) - 通用可视化工具
- [CFASample](数据工具-CFASample.md) - 数据生成
- [CFADataTest](数据探索-CFADataTest.md) - 时序数据分析

## 技术实现

CFANetGraph基于以下库实现：
- **pyvis**: 交互式网络图可视化
- **networkx**: 图结构处理
- **webbrowser**: 自动打开HTML文件

确保这些库已安装：
```bash
pip install pyvis networkx
```
