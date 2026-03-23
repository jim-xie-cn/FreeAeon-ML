# CFACommon

## 功能分类
数据工具

## 类描述
通用工具类，提供带进度条的CSV文件读取功能

## 应用场景

- 需要读取大型CSV文件并显示进度时
- 数据加载过程需要可视化进度反馈时
- 批量处理CSV文件时
        

## 方法列表


### load_csv(file_name, chunksize=1000)
带进度条读取CSV文件，支持大文件分块读取。
        

## 示例代码


from FreeAeonML.FACommon import CFACommon

# 读取CSV文件（带进度条）
df = CFACommon.load_csv('large_file.csv', chunksize=1000)
print(df.head())
        

## 参数说明


| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| file_name | str | - | CSV文件路径 |
| chunksize | int | 1000 | 每次读取的行数 |
        

## 返回值说明

返回 pandas.DataFrame 对象，包含CSV文件的所有数据。

## 注意事项


- 使用 wc -l 命令统计文件行数（仅支持Unix/Linux系统）
- 适合处理大型CSV文件
- 会自动显示tqdm进度条
        

---
*生成时间: 2026-03-23 16:10:12*
*项目: FreeAeon-ML*
