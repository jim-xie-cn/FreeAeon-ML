# CFASimilarity

## 功能分类
模型评估

## 类描述
相似度计算工具类，提供多种距离和相似度度量方法

## 应用场景

- 推荐系统：计算用户或物品之间的相似度
- 异常检测：识别与正常模式差异较大的数据
- 聚类分析：计算样本间的距离用于聚类
- 时间序列比较：比较两个时间序列的相似性
- 文本相似度：计算文档或句子的相似度
        

## 方法列表


### 主要方法

#### 1. __init__(list1, list2)
初始化，传入两个待比较的数据序列。

#### 2. Cosine()
计算余弦相似度，范围[0,1]，1表示完全相同。

#### 3. Pearson()
计算皮尔森相关系数，范围[-1,1]，1表示正相关。

#### 4. Euclidean()
计算欧式距离，值越小表示越相似。

#### 5. KSTest()
Kolmogorov-Smirnov检验，返回(统计量, p值)。

#### 6. EMD()
Earth Mover's Distance（推土机距离）。

#### 7. Manhattan()
曼哈顿距离（L1距离）。

#### 8. Minkowski()
闵可夫斯基距离（p=3）。

#### 9. Jaccard()
Jaccard相似系数，基于集合交并比。
        

## 示例代码


import numpy as np
from FreeAeonML.FAEvaluation import CFASimilarity

# 1. 准备两个数据序列
list1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
list2 = [1.1, 2.2, 3.1, 4.3, 5.2, 6.1, 7.3, 8.2, 9.1, 10.2]
list3 = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]  # 相反趋势

# 2. 计算相似度（list1 vs list2 - 相似）
sim12 = CFASimilarity(list1, list2)
print("list1 vs list2（相似）:")
print(f"  余弦相似度: {sim12.Cosine():.4f} (接近1表示相似)")
print(f"  皮尔森相关系数: {sim12.Pearson():.4f} (接近1表示正相关)")
print(f"  欧式距离: {sim12.Euclidean():.4f} (越小越相似)")
print(f"  曼哈顿距离: {sim12.Manhattan():.4f}")
print(f"  闵可夫斯基距离: {sim12.Minkowski():.4f}")
print(f"  Jaccard系数: {sim12.Jaccard():.4f}")

ks_stat, ks_pvalue = sim12.KSTest()
print(f"  KS检验: 统计量={ks_stat:.4f}, p值={ks_pvalue:.4f}")
print(f"    (p值>{0.05}表示分布相似)")

emd_dist = sim12.EMD()
print(f"  EMD距离: {emd_dist:.4f}")

# 3. 计算相似度（list1 vs list3 - 相反）
sim13 = CFASimilarity(list1, list3)
print("\nlist1 vs list3（相反）:")
print(f"  余弦相似度: {sim13.Cosine():.4f}")
print(f"  皮尔森相关系数: {sim13.Pearson():.4f} (接近-1表示负相关)")
print(f"  欧式距离: {sim13.Euclidean():.4f}")

# 4. 时间序列相似度分析
import pandas as pd
ts1 = pd.Series(np.sin(np.linspace(0, 4*np.pi, 100)))
ts2 = pd.Series(np.sin(np.linspace(0, 4*np.pi, 100)) + np.random.normal(0, 0.1, 100))
ts3 = pd.Series(np.cos(np.linspace(0, 4*np.pi, 100)))

sim_ts12 = CFASimilarity(ts1.tolist(), ts2.tolist())
sim_ts13 = CFASimilarity(ts1.tolist(), ts3.tolist())

print("\n时间序列相似度分析:")
print(f"sin vs sin+噪声:")
print(f"  皮尔森相关: {sim_ts12.Pearson():.4f}")
print(f"  欧式距离: {sim_ts12.Euclidean():.4f}")

print(f"\nsin vs cos:")
print(f"  皮尔森相关: {sim_ts13.Pearson():.4f}")
print(f"  欧式距离: {sim_ts13.Euclidean():.4f}")

# 5. 分布相似度检验
from scipy.stats import norm, expon
dist1 = norm.rvs(size=1000, loc=0, scale=1)
dist2 = norm.rvs(size=1000, loc=0.1, scale=1.1)  # 相似分布
dist3 = expon.rvs(size=1000, scale=1)  # 不同分布

sim_dist12 = CFASimilarity(dist1.tolist(), dist2.tolist())
sim_dist13 = CFASimilarity(dist1.tolist(), dist3.tolist())

ks_stat12, ks_pval12 = sim_dist12.KSTest()
ks_stat13, ks_pval13 = sim_dist13.KSTest()

print("\n分布相似度检验:")
print(f"正态 vs 正态(相似): KS统计量={ks_stat12:.4f}, p值={ks_pval12:.4f}")
print(f"正态 vs 指数(不同): KS统计量={ks_stat13:.4f}, p值={ks_pval13:.4f}")
        

## 参数说明


| 方法 | 参数 | 类型 | 说明 |
|------|------|------|------|
| __init__ | list1 | list | 第一个数据序列 |
| | list2 | list | 第二个数据序列 |
        

## 返回值说明


- **Cosine**: float，范围[0,1]
- **Pearson**: float，范围[-1,1]
- **Euclidean**: float，非负
- **KSTest**: (统计量, p值)
- **EMD**: float，非负
- **Manhattan**: float，非负
- **Minkowski**: float，非负
- **Jaccard**: float，范围[0,1]
        

## 注意事项


- 余弦相似度关注方向，不考虑大小
- 皮尔森相关系数衡量线性相关性
- 欧式距离是最常用的距离度量
- KS检验的p值>0.05表示分布相似
- EMD适用于分布比较
- Jaccard适用于集合相似度
- 不同相似度度量适用于不同场景
- 需要根据数据特点选择合适的度量方法
        

---
*生成时间: 2026-03-23 16:10:12*
*项目: FreeAeon-ML*
