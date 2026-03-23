# CFAVisualize

## 功能分类
可视化

## 类描述
数据可视化工具类，提供热图、桑基图、等高线图等多种可视化方法

## 应用场景

- 相关性分析：热图展示变量间相关系数
- 流程分析：桑基图展示数据流转过程
- 空间分布：等高线图展示二维空间中的数值分布
- 时序可视化：网络图展示时间序列演化
- 数据探索：快速理解数据的分布和关系
        

## 方法列表


### 主要方法

#### 1. show_heatmap(df, title="heatmap demo")
绘制热图，适用于展示二维矩阵数据。

#### 2. get_sankey(df_data_list, title="sankey demo")
生成桑基图，展示分类数据的流转。

#### 3. show_contour(df_data, isBlack=True, withTips=True)
显示等高线图，展示三维数据的二维投影。

#### 4. show_sequence(df_data, call_edge_info=None, file_name='pyvis.html')
可视化时序数据为网络图。
        

## 示例代码


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import webbrowser
import os
from FreeAeonML.FAVisualize import CFAVisualize

# 1. 热图可视化
# 准备相关性数据
data_heatmap = {
    '类别1': ['A1', 'A2', 'A3', 'A1', 'A2'],
    '类别2': ['B1', 'B2', 'B1', 'B2', 'B1'],
    'Count': [100, 50, 30, 80, 60]
}
df_heatmap = pd.DataFrame(data_heatmap)
pt = df_heatmap.pivot_table(
    values='Count',
    index=['类别1'],
    columns=['类别2'],
    aggfunc=np.sum,
    fill_value=0
)
print("热图数据:")
print(pt)

CFAVisualize.show_heatmap(pt, title="类别关系热图")
plt.show()

# 2. 桑基图可视化
# 准备流程数据
data_sankey = []
data_sankey.append({"阶段1": "来源A", "阶段2": "中间B1", "阶段3": "目标C1", "数量": 100})
data_sankey.append({"阶段1": "来源A", "阶段2": "中间B2", "阶段3": "目标C2", "数量": 50})
data_sankey.append({"阶段1": "来源B", "阶段2": "中间B1", "阶段3": "目标C1", "数量": 80})
data_sankey.append({"阶段1": "来源B", "阶段2": "中间B2", "阶段3": "目标C2", "数量": 70})
df_sankey = pd.DataFrame(data_sankey)

sankey_chart = CFAVisualize.get_sankey(df_sankey, title="数据流转桑基图")
html_file = './sankey_chart.html'
file_path = os.path.abspath(html_file)
sankey_chart.render(file_path)
webbrowser.open(f"file://{file_path}")
print(f"桑基图已保存到: {file_path}")

# 3. 等高线图可视化
# 准备三维数据
data_contour = []
for i in range(20):
    for j in range(20):
        # 创建一个山峰形状
        x_norm = (i - 10) / 10
        y_norm = (j - 10) / 10
        value = np.exp(-(x_norm**2 + y_norm**2))
        data_contour.append({"x": i, "y": j, "value": value})

df_contour = pd.DataFrame(data_contour)
print("\n等高线数据样例:")
print(df_contour.head())

plt.figure(figsize=(10, 8))
category_map = CFAVisualize.show_contour(df_contour, isBlack=False, withTips=True)
plt.title("数值分布等高线图")
plt.xlabel("X坐标")
plt.ylabel("Y坐标")
plt.show()

# 4. 时序网络图可视化
# 准备时序数据
data_sequence = []
for i in range(10):
    tmp = {
        'label': f"节点{i}",
        'value': np.random.randint(10, 100),
        'title': f"时间点{i}\n数值: {np.random.randint(10, 100)}",
        'color': '#' + ''.join([np.random.choice(list('0123456789ABCDEF')) for _ in range(6)])
    }
    data_sequence.append(tmp)

df_sequence = pd.DataFrame(data_sequence)
html_file = './sequence_network.html'
CFAVisualize.show_sequence(df_sequence, file_name=html_file)
print(f"\n时序网络图已保存到: {os.path.abspath(html_file)}")

# 5. 分类变量等高线
# 包含分类变量的数据
data_cat = []
categories_x = ['低', '中', '高']
categories_y = ['差', '良', '优']
for i, cat_x in enumerate(categories_x):
    for j, cat_y in enumerate(categories_y):
        value = (i + 1) * (j + 1) * 10
        data_cat.append({"等级": cat_x, "质量": cat_y, "得分": value})

df_cat = pd.DataFrame(data_cat)
plt.figure(figsize=(8, 6))
category_map = CFAVisualize.show_contour(df_cat, isBlack=True, withTips=True)
plt.title("分类变量等高线图")
if category_map:
    print("\n分类映射:")
    for key, mapping in category_map.items():
        print(f"{key}: {mapping}")
plt.show()

# 6. 多阶段桑基图（使用列表）
df_stage1 = pd.DataFrame({
    '来源': ['A', 'B'],
    '中间1': ['M1', 'M2'],
    '数量': [100, 80]
})
df_stage2 = pd.DataFrame({
    '中间1': ['M1', 'M2'],
    '目标': ['T1', 'T2'],
    '数量': [100, 80]
})

sankey_multi = CFAVisualize.get_sankey([df_stage1, df_stage2], title="多阶段流程")
html_file = './sankey_multi.html'
sankey_multi.render(os.path.abspath(html_file))
print(f"\n多阶段桑基图已保存到: {os.path.abspath(html_file)}")
        

## 参数说明


| 方法 | 参数 | 类型 | 说明 |
|------|------|------|------|
| show_heatmap | df | pd.DataFrame | 二维矩阵数据 |
| | title | str | 图表标题 |
| get_sankey | df_data_list | DataFrame/list | 单个DataFrame或DataFrame列表 |
| | title | str | 图表标题 |
| show_contour | df_data | pd.DataFrame | 三列数据(x, y, value) |
| | isBlack | bool | 是否黑白显示 |
| | withTips | bool | 是否显示数值标签 |
| show_sequence | df_data | pd.DataFrame | 时序数据（含label、value、title） |
| | call_edge_info | callable | 自定义边信息计算函数 |
| | file_name | str | 输出HTML文件名 |
        

## 返回值说明


- **show_heatmap**: 无返回值（显示图表）
- **get_sankey**: Sankey图表对象
- **show_contour**: 分类映射字典
- **show_sequence**: 无返回值（生成HTML文件）
        

## 注意事项


- 热图适合展示相关性矩阵或混淆矩阵
- 桑基图节点名称不能重复
- 桑基图的source和target必须在nodes中定义
- 等高线图可以处理分类变量
- 时序网络图会自动在浏览器中打开
- 建议在数据量适中时使用可视化
- 可视化前应确保数据格式正确
        

---
*生成时间: 2026-03-23 16:10:12*
*项目: FreeAeon-ML*
