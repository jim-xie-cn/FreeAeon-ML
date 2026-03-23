# CFANetGraph

## 功能分类
可视化

## 类描述
网络图可视化工具类，用于创建和展示复杂的节点-边关系网络

## 应用场景

- 时序数据可视化：展示时间序列的演化过程
- 关系网络分析：展示实体间的关联关系
- 流程图绘制：展示业务流程或数据流转
- 依赖关系展示：展示组件或模块间的依赖
- 知识图谱：展示概念或实体间的关系
        

## 方法列表


### 主要方法

#### 1. __init__(directed=True)
初始化网络图，指定是否为有向图。

#### 2. Add_Node(label, value, title, group=None, size=None, color='#00ff1e', node_id=None)
添加节点到网络图。

#### 3. Add_Edge(from_id, to_id, weight, label)
添加边连接两个节点。

#### 4. Add(from_node, to_node, title=None, weight=None)
添加节点对及其连接（自动创建节点和边）。

#### 5. Show(file_name='pyvis.html')
生成并保存网络图为HTML文件。

#### 6. Show_Series(df_in, call_edge_info=None, file_name='pyvis.html')
静态方法，可视化时序数据为网络图。
        

## 示例代码


import pandas as pd
import numpy as np
import webbrowser
import os
from FreeAeonML.FAVisualize import CFANetGraph

# 1. 创建简单的有向网络图
net = CFANetGraph(directed=True)

# 添加节点
node1_id = net.Add_Node(
    label="开始",
    value=100,
    title="流程开始节点",
    color="#FF6B6B",
    size=30
)
node2_id = net.Add_Node(
    label="处理",
    value=80,
    title="数据处理节点",
    color="#4ECDC4",
    size=25
)
node3_id = net.Add_Node(
    label="结束",
    value=60,
    title="流程结束节点",
    color="#95E1D3",
    size=20
)

# 添加边
net.Add_Edge(node1_id, node2_id, weight=5, label="数据传递")
net.Add_Edge(node2_id, node3_id, weight=3, label="结果输出")

# 显示网络图
html_file = './network_simple.html'
net.Show(html_file)
print(f"简单网络图已保存到: {os.path.abspath(html_file)}")
webbrowser.open(f"file://{os.path.abspath(html_file)}")

# 2. 使用Add方法快速构建网络
net2 = CFANetGraph(directed=False)

# 定义节点字典
from_node = {
    'id': 'A',
    'label': '节点A',
    'value': 50,
    'title': '这是节点A',
    'color': '#FF6B6B'
}
to_node = {
    'id': 'B',
    'label': '节点B',
    'value': 40,
    'title': '这是节点B',
    'color': '#4ECDC4'
}

# 添加节点对和连接
net2.Add(from_node, to_node, title="A到B", weight=10)

# 继续添加更多节点
to_node2 = {
    'id': 'C',
    'label': '节点C',
    'value': 30,
    'title': '这是节点C',
    'color': '#95E1D3'
}
net2.Add(to_node, to_node2, title="B到C", weight=8)

html_file = './network_quick.html'
net2.Show(html_file)
print(f"\n快速构建网络图已保存到: {os.path.abspath(html_file)}")

# 3. 时序数据网络可视化
# 准备时序数据
dates = pd.date_range('2024-01-01', periods=10, freq='D')
data_series = []
for i, date in enumerate(dates):
    tmp = {
        'label': date.strftime('%Y-%m-%d'),
        'value': 100 + np.random.randint(-20, 20),
        'title': f"日期: {date.strftime('%Y-%m-%d')}\n数值: {100 + np.random.randint(-20, 20)}",
        'color': '#' + ''.join([np.random.choice('0123456789ABCDEF') for _ in range(6)])
    }
    data_series.append(tmp)

df_series = pd.DataFrame(data_series)

# 自定义边信息计算函数
def custom_edge_info(from_node, to_node):
    weight = abs(from_node['value'] - to_node['value'])
    title = f"变化: {to_node['value'] - from_node['value']}"
    return weight, title

html_file = './network_series.html'
CFANetGraph.Show_Series(df_series, call_edge_info=custom_edge_info, file_name=html_file)
print(f"\n时序网络图已保存到: {os.path.abspath(html_file)}")

# 4. 复杂关系网络
net3 = CFANetGraph(directed=True)

# 创建多层网络结构
layers = {
    'input': ['I1', 'I2', 'I3'],
    'hidden': ['H1', 'H2'],
    'output': ['O1']
}

node_positions = {}
colors = {
    'input': '#FF6B6B',
    'hidden': '#4ECDC4',
    'output': '#95E1D3'
}

# 添加所有节点
for layer_name, nodes in layers.items():
    for node_name in nodes:
        node_id = net3.Add_Node(
            label=node_name,
            value=np.random.randint(10, 50),
            title=f"{layer_name}层节点: {node_name}",
            color=colors[layer_name],
            node_id=node_name
        )
        node_positions[node_name] = node_id

# 添加层间连接
for input_node in layers['input']:
    for hidden_node in layers['hidden']:
        net3.Add_Edge(
            input_node,
            hidden_node,
            weight=np.random.randint(1, 10),
            label=f"{np.random.randint(1, 10)}"
        )

for hidden_node in layers['hidden']:
    for output_node in layers['output']:
        net3.Add_Edge(
            hidden_node,
            output_node,
            weight=np.random.randint(1, 10),
            label=f"{np.random.randint(1, 10)}"
        )

html_file = './network_complex.html'
net3.Show(html_file)
print(f"\n复杂网络图已保存到: {os.path.abspath(html_file)}")

# 5. 带分组的网络图
net4 = CFANetGraph(directed=False)

# 创建三个组的节点
for group_id in range(3):
    group_name = f"组{group_id + 1}"
    for i in range(5):
        node_label = f"节点{group_id}-{i}"
        net4.Add_Node(
            label=node_label,
            value=np.random.randint(10, 50),
            title=f"{group_name}的{node_label}",
            group=group_name,
            color=['#FF6B6B', '#4ECDC4', '#95E1D3'][group_id]
        )

# 添加组内和组间连接
for i in range(net4.m_nodes.__len__() - 1):
    if np.random.random() > 0.7:  # 随机连接
        net4.Add_Edge(
            net4.m_nodes[i]['id'],
            net4.m_nodes[i+1]['id'],
            weight=np.random.randint(1, 10),
            label=""
        )

html_file = './network_grouped.html'
net4.Show(html_file)
print(f"\n分组网络图已保存到: {os.path.abspath(html_file)}")
        

## 参数说明


| 方法 | 参数 | 类型 | 说明 |
|------|------|------|------|
| __init__ | directed | bool | True为有向图，False为无向图 |
| Add_Node | label | str | 节点标签 |
| | value | int/float | 节点值（影响大小） |
| | title | str | 鼠标悬停提示 |
| | group | str | 分组名称 |
| | size | int | 节点大小 |
| | color | str | 节点颜色（十六进制） |
| | node_id | str/int | 节点ID，None时自动生成 |
| Add_Edge | from_id | str/int | 起始节点ID |
| | to_id | str/int | 目标节点ID |
| | weight | int/float | 边的权重（影响粗细） |
| | label | str | 边的标签 |
| Add | from_node | dict | 起始节点字典 |
| | to_node | dict | 目标节点字典 |
| | title | str | 边的提示信息 |
| | weight | int/float | 边的权重 |
| Show | file_name | str | 输出HTML文件名 |
| Show_Series | df_in | pd.DataFrame | 时序数据 |
| | call_edge_info | callable | 边信息计算函数 |
| | file_name | str | 输出HTML文件名 |
        

## 返回值说明


- **Add_Node**: 节点ID
- **Add_Edge**: 无返回值
- **Add**: (起始节点ID, 目标节点ID)
- **Show**: 无返回值（生成HTML文件）
- **Show_Series**: 无返回值（生成HTML文件）
        

## 注意事项


- 节点ID必须唯一，重复ID不会添加新节点
- 边的权重影响边的粗细
- 节点的value影响节点大小
- 颜色使用十六进制格式（如#FF6B6B）
- 时序数据需要包含label、value、title字段
- 生成的HTML文件可在浏览器中交互
- 网络图支持拖拽、缩放等交互操作
- 大规模网络（>1000节点）可能影响性能
        

---
*生成时间: 2026-03-23 16:10:12*
*项目: FreeAeon-ML*
