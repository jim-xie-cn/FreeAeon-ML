# CFACommon - 通用工具类

## 应用场景

CFACommon类提供常用的工具方法,主要应用于:

- 大文件高效加载
- 带进度条的数据读取
- 数据IO操作

## 安装依赖

```bash
pip install FreeAeon-ML
```

## 类说明

CFACommon是静态方法工具类,无需实例化即可使用。

## 方法详解

### load_csv - 带进度条加载CSV

```python
@staticmethod
def load_csv(file_name, chunksize=1000)
```

分块加载大型CSV文件并显示进度条。

**参数**:
- file_name: CSV文件路径
- chunksize: 每块行数(默认1000)

**返回**: pandas DataFrame

**示例**:
```python
from FreeAeonML.FACommon import CFACommon

# 加载大文件
df = CFACommon.load_csv('large_file.csv', chunksize=5000)
print(df.shape)
print(df.head())
```

## 完整示例

```python
from FreeAeonML.FACommon import CFACommon
import pandas as pd
import numpy as np

# 创建测试文件
print("创建测试CSV文件...")
df_test = pd.DataFrame({
    'id': range(50000),
    'value': np.random.randn(50000),
    'category': np.random.choice(['A', 'B', 'C'], 50000)
})
df_test.to_csv('test_large.csv', index=False)

# 使用load_csv加载
print("\n使用CFACommon.load_csv加载...")
df = CFACommon.load_csv('test_large.csv', chunksize=10000)

print(f"\n加载完成!")
print(f"数据形状: {df.shape}")
print(f"内存占用: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
print(f"\n前5行数据:")
print(df.head())
```

## 注意事项

1. **chunksize选择**: 
   - 过小:进度条更新频繁,速度慢
   - 过大:内存占用高,进度条不准确
   - 推荐:10000-50000

2. **大文件加载**: 
   - 适用于GB级CSV文件
   - 自动使用tqdm显示进度

3. **内存管理**:
   - 分块读取节省内存
   - 避免一次性加载超大文件

## 使用场景

1. **日志分析**: 加载大型日志文件
2. **数据迁移**: 读取大型数据导出文件
3. **批量处理**: 处理海量CSV数据

## 相关类链接

- [CFASample](./数据工具-CFASample.md) - 样本数据生成
