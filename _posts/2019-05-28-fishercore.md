---
layout:     post
title: Mathematics of a Model Predicting Happiness
subtitle: Fisher假设检验方法用于多分类问题评分的思考
date:       2019-05-28
author:     Loopy
header-img: img/home-bg-geek.jpg
catalog: true
tags:
    - Data

---

<script type="text/x-mathjax-config">
  MathJax.Hub.Config({
    tex2jax: {
      inlineMath: [ ['$','$'], ['\\(','\\)'] ],
      processEscapes: true
    }
  });
  </script>
<script type="text/javascript" async src="//cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>


> 也可参见本文的[jupyter发布版本](http://file.loopy.tech/release/FisherScore.html),[Github仓库](https://github.com/loopyme/AliTianChi/blob/master/多分类_幸福度/Fisher假设检验方法用于多分类问题评分的思考.ipynb),或可在线运行的平台:[online-playground](http://jupyter.loopy.tech:8888/notebooks/Fisher假设检验方法用于多分类问题评分的思考.ipynb),[天池实验室](https://tianchi.aliyun.com/notebook-ai/detail?spm=5176.12281915.0.0.30756d844eWfcf&postId=58920)

# Fisher假设检验方法用于多分类问题评分的思考

## 一. 问题的发现
在阿里天池的[幸福感预测比赛](https://tianchi.aliyun.com/competition/entrance/231702/introduction?spm=5176.12281949.1003.6.493e2448kC12t6)（实质上是个多分类预测问题）中，使用到了均方误差（MSE）评分的办法。我发现MSE虽然能表征预测结果的有效性，但实际上存在一个漏洞。

考虑一个测试集,将其2k个样本的结果作为空间$F^{2000}$下的一个基准点$Y$.在提交了十个结果以后,用这十个点确定一个平面(线性子空间)$F^{10}$,再从$Y$向$F^{10}$作投影,投影点$P$即为已知空间$F^{10}$中最优的一点。取$P$会导致MSE急剧下降，原理在于,这个算法使用测试集得分对结果的有效性进行校正(实际上会达到最优),而只依赖于结果的无偏性。基于这个思路，我已实现了算法，并证明这个漏洞可以被利用，在上文提到的这场比赛中获得了线上MSE=0.1971的成绩，（如果有效的话）在比赛排名第2，与其他0.4+的得分相比，属于异常数据。并由于疏忽，于2019-05-12将结果提交到了比赛平台，目前正在联系平台寻找[删除成绩的办法](https://tianchi.aliyun.com/forum/issueDetail?spm=5176.12586969.1002.36.52a16cd0kzuT2n&postId=58688)。

由此可见，在数据挖掘的离线赛中，均方误差（MSE）评分法存在较大的漏洞。

## 二. Fisher假设检验方法的思考
为了填补这个漏洞，恰逢我的概率论老师荣sir要求我们阅读罗纳德.费舍尔著作的"6 Mathematics of a Lady Tasting Tea"，我开始思考Fisher假设检验的思路能否用于多分类预测问题比赛的评分？

仿照费舍尔的思路，我完成了"Mathematics of a Model Predicting Happiness"的思考。
## 三.Mathematics of a Model Predicting Happiness
### 实验说明
一个Xgboost模型声称，通过阅读一个人关于生活的问卷，它可以辨别出那个人的幸福程度。我们将考虑设计一个实验，通过这个实验来验证这一论断。为此，我们首先提出一种简单的实验形式，研究这一实验中的限制和特点。

我们的实验包括1000条样本数据，然后随机地把它们呈现给一个Xgboost模型进行判断。它已被告知提前测试将包括什么,也就是说,它已通过预先学习（fit）另外的9000条样本数据，知晓各特征之间的相关关系，知道它应通过哪些输入来判断Happiness的分类结果（在这个数据集中Happiness被分为0到5，共六个等级）



```python
# 实验准备
import pandas as pd
from xgboost import XGBClassifier
from scipy.special import comb
from math import *
from sklearn.metrics import mean_squared_error,accuracy_score
import warnings
warnings.filterwarnings("ignore")

# 准备数据集:我简化的使用[0:1000]做测试集,[1000:]做训练集
data = pd.read_csv('./data/happiness_train_complete.csv',encoding='gbk').drop(['survey_time',"property_other","invest_other","edu_other"],axis=1)
data = data.fillna(-8)[data['happiness']>=0]

# 准备模型
model = XGBClassifier().fit(data.drop(['happiness'],axis=1).iloc[1000:],data["happiness"].iloc[1000:])

# 分割数据集
data = data.iloc[:1000]

# 使用模型作出预测
pred = pd.DataFrame([model.predict(data.drop(['happiness'],axis=1)),data['happiness']],index=['pred','true']).T

# 数据集描述
count = [[],[],[],[]]
for i in range(1,6):
    count[0].append(data[data["happiness"]==i]["happiness"].count())
    count[1].append(pred[pred['pred']==i]['pred'].count())
    count[2].append(pred[pred['true']==i][pred['pred']==pred['true']]['pred'].count())
    count[3].append(pred[pred['true']==i][pred['pred']!=pred['true']]['pred'].count())

# 使用模型作出预测
pd.DataFrame([model.predict(data.drop(['happiness'],axis=1)),data['happiness']],index=['pred','true'])

#输出
count_log = pd.DataFrame(count,index = ["真实频数","预测频数","正确判断数","错误判断数"],columns = [1,2,3,4,5])
count_log
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>真实频数</th>
      <td>17</td>
      <td>53</td>
      <td>151</td>
      <td>604</td>
      <td>175</td>
    </tr>
    <tr>
      <th>预测频数</th>
      <td>4</td>
      <td>37</td>
      <td>20</td>
      <td>892</td>
      <td>47</td>
    </tr>
    <tr>
      <th>正确判断数</th>
      <td>3</td>
      <td>14</td>
      <td>10</td>
      <td>575</td>
      <td>28</td>
    </tr>
    <tr>
      <th>错误判断数</th>
      <td>14</td>
      <td>39</td>
      <td>141</td>
      <td>29</td>
      <td>147</td>
    </tr>
  </tbody>
</table>
</div>



### 解释及其依据

> 费舍尔在他的文章中提出:频率分布适合一个分类结果纯粹是偶然的.如果没有辨别能力,实验的结果将完全由随机的概率决定

在考虑任何设想的实验设计是否合适时，总是需要预测实验的可能结果，并决定对每一个结果应作何种解释。此外，我们必须知道这种解释要用什么论据来支持。在目前的情况下，我们可以这样说:从2000个样本中选出其中的17个Happiness=0的对象有$C_{1000}^{17}$种方法。对一种没有分类效果的模型来说,它正确选出这17个对象的概率是$\dfrac{1}{C_{1000}^{17}}\approx4.08e^{-37}$.


```python
1/comb(1000, 17)

```




    4.078121130799551e-37



### 零假设
 - $H_0$:这个Xgboost模型不能通过阅读一个人关于生活的问卷，辨别出那个人的幸福程度。

### 检验

> 费舍尔在他的文章中,主要通过当前观察到的事件的极端程度(概率),来对事件空间的分布进行预测,所以既然要考察当前观察到的事件有多么极端，那就不仅要知道该事件个体点发生的概率，还要知道它在整个事件空间中所处的位置，即比这个事件更极端的事件空间的总概率,考虑到事件是离散的,可以用$\sum$来解决,这里我暂时将它称为极限概率

1. 如果这个xgboost模型选出了17个Happiness=0的对象:上面我们计算出误打误撞获得正确结果的概率是一个极小的值,这使得如果这个模型能精确的选出这17个Happiness=0的对象,那么我们就有很大的把握说它能够辨别出某个人的幸福程度是否为1.但如果,这个模型只选出了3个正确答案.那么误打误撞选出3个正确答案的可能性则为:$\dfrac{C_{17}^3\times C_{983}^{14}}{C_{1000}^{17}}\approx0.00228$,也就能计算出极限概率$\sum_{i=3}^{17}{\dfrac{C_{17}^i\times C_{983}^{17-i}}{C_{1000}^{17}}}\approx0.0024$,这就是说只有0.0024的可能性,在$H_1$成立时,事件会落到这个极端事件空间里

2. 可是,这个xgboost模型只选出了4个Happiness=0的对象:只选出了4个并正确3个的单事件概率:$\dfrac{C_{4}^3\times C_{996}^{1}}{C_{1000}^{4}}\approx9.62e^{-8}$,极限概率:$\sum_{i=3}^{4}{\dfrac{C_{4}^i\times C_{996}^{4-i}}{C_{1000}^{4}}}\approx9.62e^{-8}$

这两个极限概率都很小,能够支撑起$H_1$的判断,但如果需要建立一个量化的评判标准,选那个比较好?




```python
# 极限概率计算函数
def calculate_p(count_all,count_try,count_true):
    p = 0
    for i in range(count_true,count_try+1):
        p+=comb(count_try, i)*comb(count_all-count_try, count_try-i)/comb(count_all, count_try)
    return p
calculate_p(1000,17,3),calculate_p(1000,4,3)
```




    (0.002399205081394864, 9.621623963648032e-08)



### 评判标准

我考虑到:如果采用检验2中的方法,实质上每个分类的事件空间相互挤压,发生了较大变形,可能会造成频率信息的失真,所以检验1中的计算方法更加合理.于是我按照检验1中的计算方法，考虑六个分类，将这个评判的量化标准表示为:
$${pscore}=-\sum_{i=1}^{S_{cat}}ln({calculate_p}(S_{sample},S_i,S_i^*))$$
$${calculate_p}(S_{sample},S_i,S_i^*)=\sum_{i=S_i^*}^{S_i}{\dfrac{C_{S_i}^i\times C_{S_{sample}-S_i}^{S_i-i}}{C_{S_{sample}}^{S_i}}}$$

联立起来就是：
$${pscore}=-\sum_{i=1}^{S_{cat}}ln(\sum_{i=S_i^*}^{S_i}{\dfrac{C_{S_i}^i\times C_{S_{sample}-S_i}^{S_i-i}}{C_{S_{sample}}^{S_i}}})$$

|符号|意义|
| -------- | -------- |
| $pscore$ | 模型得分 |
| $S_{cat}$ | 分类总数 |
| $ln$ | 自然对数函数 |
| ${calculate_p}$ | 极限概率计算函数 |
| $S_{sample}$ | 样本容量 |
| $S_i$ | (真实值中)第i类中的正确分类个数 |
| $S_i^*$ | (真实值中)第i类个数 |

这个得分表征了不具预测能力的模型获得这个结果及更优结果的可能性大小(或者预测的事件及更优事件占全体事件空间的比例),其值为一个正数,越大表示越优秀.

也就是说,这样一个模型能把事件空间$F$的所有可能性坍缩到$e^{-score}$倍大小的空间中.


```python
def score(y_pred,y_true):
    limit = [y_true.min(),y_true.max()]
    s_sample = y_true.count()
    s_cat = limit[1]-limit[0]
    score = 0

    data = pd.DataFrame([y_pred,y_true],index=['pred','true']).T

    for i in range(limit[0],limit[1]+1):
        s_i_ = data[data['true']==i]['true'].count()
        s_i =  data[data['true']==i][data['pred']==data['true']]['pred'].count()
        score += log(calculate_p(s_sample,s_i_,s_i))
    return -score
```


```python
score(pred['pred'],pred['true']),score(pred['true'],pred['true'])
```




    (474.99645867621354, 1837.2800383353233)




```python
%%timeit
score(pred['pred'],pred['true'])
```

    61.1 ms ± 575 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)



```python
%%timeit
mean_squared_error(pred['pred'],pred['true'])
```

    230 µs ± 5.7 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)



```python
%%timeit
accuracy_score(pred['pred'],pred['true'])
```

    377 µs ± 51.1 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)


计算出这个Xgboost模型的score是474,由此,我们可以拒绝零假设$H_0$，即认为这个模型具有预测幸福度的能力.同时,我们也能计算出最优情况下score为1837.

p-score是很直观的，看下面的Venn图，是一个p-score=1和p-score=2的预测结果的示意图。

预测结果，及比其更优的预测结果组成了预测事件空间；而全体事件，则组成了事件空间。p-score就表征了模型预测导致的事件空间可能性坍缩比例。由此可知，p-score越大，预测事件空间越小，就越精准。因此，p-score能量化的表现预测效果：我们能够明确的说p-score=2的模型比p-score=1的模型预测效果好$e^{2-1}=e$倍(自然对数)。


```python
# 绘图
from matplotlib import pyplot as plt
from matplotlib_venn import venn2

fig=plt.figure(figsize=(20,10))

ax = fig.add_subplot(221)
plt.title("P-score=1示意图")
v = venn2(subsets=(e, 0, 1), set_labels=('', ''))
v.get_label_by_id('10').set_text('事件空间')
v.get_label_by_id('01').set_text('')
v.get_label_by_id('11').set_text('预测事件空间')

ax2 = fig.add_subplot(211)
plt.title("P-score=2示意图")
v = venn2(subsets=(e**2, 0, 1), set_labels=('', ''))
v.get_label_by_id('10').set_text('事件空间')
v.get_label_by_id('01').set_text('')
v.get_label_by_id('11').set_text('预测事件空间')
fig.show()

```
[!1.png](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAjwAAAEYCAYAAABY2iwnAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDIuMi4yLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvhp/UCwAAIABJREFUeJzt3Xl03Xd95//n5266WmzJtrzG+57YWUnixGSBQGKyEApMB0o7HWgpTKdQoEOH9vArnemZ6ULLWkpheshAKd0GWkJISAgJ2RyyO5vjJd4t77ZkS9bV1d0+vz8+V7EsS9Z27/18v9/7epyjY1vrW/K9H70+u7HWIiIiIhJlMd8FiIiIiFSbAo+IiIhEngKPiIiIRJ4Cj4iIiESeAo+IiIhEngKPTJoxpsEYc85jyRgTM8akRvnYXzXGfK561YmIjEztV/0w2pZeG8aYDwLfBE4DOeAr1to/91rUOBlj3gZ80Fr7n4a8/mFgAVAc8iExYJO19n2D3vfPgWuAbPlV04HZwJbyv5uBz1tr7ym//0+B1UBmhLJagI2Dv4aIVFbY2y9jzLuALwMNwP9nrb1r0NvUftWJhO8C6sy/W2vfb4yZBTxujHnEWvuU76LGwhjzIHA98MOhb7PW3jTWz2Ot/YPy57sQ+Ji19h3lf98FfN1a+9yQD8kDv26tfcQYczvwLmvtRwbV9UvAfxjv9yMi4xbK9ssY0wj8HXATLqy9YIz5obW2E9R+1RMFHg+stUeNMfcCNwKBaTCMMf+IaxQG+xdr7SestTeXe3nvGPT+BkgBOXueocLysHABWAL8Na7RiQNLjTEDAWoD0G6MseXP+UFr7RFcg4ExphX4ItBjjHkOuHRQLYWJf9ciMh5ha7+Au4Bj1tpXy+/XAyw3xjyL2q+6osDjV2mkNxhj/gj4nfI//9ha+83y638V+F9AI/AFa+1fll//a+XXm/L7f7v8+keArwG/Bkwd6M0YYz4EfK78ef7IWvt31toPjLP+2cBGIO+yDwCt5c95eND7NQDvwg37/gbQD4zUwJjy+x8f9LoY8AOgG/iH8sdeaK19qNxDEpHaC0X7ZYxJ4kanMcbMA2YA+1H7VXcUeDwwxlwA3AHsNcZ8Zsib/wn4Y+CPgLlAEvgG8M3yMOqfA+txT7qXjDF3455Qf1F+fQl40hjzvLX2lfLn/FPg08Cj5a+/Bvg94Ory53/WGPOjco9kzKy1h4FlQ763DwLXWWs/PMKHHTbGfA3XEzs65G3zgG9Za/9iyOtL5fp7gF/gGpW3jKdWEamMsLVf1to80GmMiQP/B/hba+2h8pvVftURBZ7aepcx5jDQC3zRWvs3w71TecfAVlwj8ADwq+U3vR34sbV2f/n95uOeTB8D7rHW7i2//t+AW4CBBuMua+2PBn2Jm4Clg97eCKwyxnwVeNuQcv7JWvvx831T5Ybs+8CbB72uuVz7b1prtw35kBzwIvDykNevK79tqNnAH+Ier18DXgK+aoyZhhuuFpHqC237Va7pO+Wv93tD6lX7VScUeGrrbmvt+0d7J2ttyRhzNa4XcDvwv4wxFw/zrm8D9g582JC3mUF/HzrPboC/t9b+NoAxpg3IWmsfG/1bGNafAd+31p4cGBq21vYaY/4fcJ8x5mpr7YlB758CLsP1iAabN0ytAEeAD1lrDxpjfg/XkH4KeBq4AbhygnWLyNiFuf36n8BM4E5r7dDdWGq/6oTO4QkgY8xK4DHc/PJncU+k6cDDwB3GmAXlJ/nf4LZB/gx4pzFmYXm4+d243slIfg7cZoyZV/48LwKrJljrR3E9mD8Z+jZr7VfK38P3Br3/VOATwK3Ae3G9u7+y1r7FWrsSuN8Y0zL4S5Q/10FjzF8Ad+KGpV8e2GXB2Y2jiHgUtPar/Dl/DfiP1tr+IW9T+1VHNMITQNba7caYh4Cd5Vd9rbxe5rAx5rPA47j/uy9Ya18AMMb8Aa6RMcDnBs1/D/f5XzHG/AnwJK638gVr7UvjrbM8THw9bkdCsTxH3jLk3T7O2T2YzwMP4XZ4PFZ+ebcx5pXy69qB/2GMeU9550RyUN2fMcZcBPxnYAVwoPwmPY5FAiKA7dd7cOuJdg5anPxJ4B9R+1VXdPCgVEx5q+rNwEcGdlkMeXsKeATXyPwu8NvAwfKbL8ANK/9heSHk18rDyjq4S0SqTu1X9CnwSMUYY+YCPdba0xX8nNOAXmvtcIsBRUQqQu1X9CnwiIiISORp0bKIiIhEngKPiIiIRJ4Cj4iIiESeAo+IiIhEngKPiIiIRJ4Cj4iIiESeAo+IiIhEngKPiIiIRJ4Cj4iIiESeAo+IiIhEngKPiIiIRJ4Cj4iIiESeAo+IiIhEngKPiIiIRJ4Cj4iIiESeAo+IiIhEngKPiIiIRJ4Cj4iIiESeAo+IiIhEngKPiIiIRJ4Cj4iIiERewncB4pExBkgC8fJLrPynGfRetvxSAvqwNl/rMkVEhmVMAteGDbRdsfKL4UzbNfCSx7Vh1k+x4pvR/31EGRMDmkd5aWL8o3wFIAv0lV8y5X8P/NkLdGFtdvLfhIjUJdcZa8S1UcO1WwN/T43zM1ugH9deDW7DBv/9NHBKwSh6FHiiwJgk0A7MBGaV/5zitSYXfE4Ax8t/nsDabr8liUjguHDThmu3Bl5m4EZsfCkAnbj2a6AN68TaoseaZJIUeMLGjdzM4Oxw08bZ01BBlcM1IieAY8ABrO31W5KI1JQxUzk73LTjpqWCrgSc5EwAOgoc1UhQeCjwhIExU4DFwCJgDtFabH4C2A/sA46o8RCJGLfOZj6uDVsIpL3WU1lZoAPYC3Rgbb/neuQ8FHiCyph2XMBZjBvRqQf9uMZjH7Bf64BEQsqYRlz7tQi4gPrYIFMCjuDar31Y2+W5HhlCgSco3Dz2XFzAWQy0+CwnACxu2msP8LqmvkQCzpg2znTSZhGOafZq6sGFn91Ye9B3MaLA458x04ALgeVEa6i3kixwANgG7NHCQZGAMCYFrMC1YdM9VxNk3bj2a7s6b/4o8PhgTBxYimsk5niuJmz6gdeB17D2pO9iROqSMbNw7dcy6mO6qlIsbtr+Ndy0l34B15ACTy0Z0wSswTUUGs2ZvAPAZmCvGg6RKnM7RJcBF+N2Vsnk9OCCz1Ytdq4NBZ5acAuQL8Y1FlHaYRUUp4GXgC1YW/JdjEikGJPGddLW4A79k8oqADuATVjb47uYKFPgqSYXdK7GbcmU6jsNPI+bJ9cDW2Qy3Pqcy4C1aNqqFkrAVuAFrM34LiaKFHiqwZ2bcxVuIbLU3ingOazd6bsQkdBxU1drgMvR1LsPBdxU/Yua6qosBZ5KckO/l+MaC01d+deJCz57fBciEgrGrACuxP/VNOJOpn8ZeEWXNleGAk8luJNE1+KGf8d7mZ1U31Fc8OnwXYhIIBkzH1hH/RxyGiZZYBNuZ6qO5JgEBZ7JcIcFrsT1iJo9VyOj2ws8rvlxkTK3znAd7jRkCbbTwBNYu893IWGlwDNR7sDAt+Auv5PwyAFPYe1W34WIeONGpa/CjUzX+4nIYbMDeFJX74yfAs94uVGdS3CjOnHP1cjEHQQew9pu34WI1JQxs3GdtVbPlcjE9QEbsXaX70LCRIFnPIxpxTUUsz1XIpVRAJ7DLQrUE0GizZ3wfiWuw6ZRnWjYjQs+mqYfAwWesTJmLe5MHZ1HET1HgUd1u7FEljEzcZ21aZ4rkcrrB36Btdt9FxJ0CjyjcWfq3AjM812KVFUJeAF32qmeFBIN7kydK3A7SHVURrTtx03T63LSESjwnI8xq4FrgaTvUqRm9gEPY23OdyEik+I2VtyEtprXkyzwINYe8l1IECnwDMf1iq4HVvkuRbzoBh7QFJeEljGLgbeizlo9KuF2or7qu5CgUeAZyp2WfAswx3cp4lUet65HuyAkXIy5HLflXOrbNty5PTqssEyBZzBjpgMb0LHqcsaLwLNa1yOB53Zh3Yju8JMzjuKmuLSuBwWeM4xZhJvv1hCwDNUBPKSL/CSwjGnCjUzP8l2KBE4fLvQc9l2Ibwo8AMZchhsC1tkUMpJu4KdY2+m7EJGzuC3nt6DrbWRkJdzpzK/5LsSn+g48bgj4etx9WCKjyQP3aweEBIYxy3DTWDofTMbiVax90ncRvtRv4DEmBbwDLU6W8Snihod1gZ/4ZcyluIs/RcZjG+68nrr75V+fgceYBuA2dPGnTEwJ+DnW7vRdiNQpY67AXRMhMhG7cOeNlXwXUkv1F3jctvPbgHbfpUioWeBx3bouNWfMVcDlvsuQ0NuHG62um23r9RV4XNi5A5juuxSJjMcUeqRmjFkHXOq7DImMDtwhq3UReurnbhU3jaWwI5V2Q/kKEpHqMuZqFHaksuYDt5RvF4i8uvgmywuUb0NhR6rjBozRNSRSPW7NzmW+y5BIWkCdhJ7If4MYk8DtxtICZammGzFmie8iJIKMuQQtUJbqWgi8HWMifRZdtAOPS6wb0NZzqY23YowWw0vluOnSa3yXIXVhMRE/5iDagQfeDFzguwipG2400RideCuTZ8xc4DrfZUhduSTK0/PRDTzGXARc6LsMqTtNuNCjk29l4oxpAW4mym20BNX15bAdOdF8Mrn/rPW+y5C6NQN4W9Tnw6VKXFjeAKR9lyJ1KQbcjDFTfRdSadELPOoZSTAsIuLz4VI1b8GFZhFf0riR6pTvQiopWqHA9YxuQT0jCYZLdEaPjIvbfr7UdxkiQBsR27kVrcDjekbaJSNBch3GzPNdhISAMYvQ9nMJlvlEaHlIdAKPMZejnpEETwy3nqfRdyESYMZMA27yXYbIMNZEZedWNAKPMQtQz0iCqxG43ncRElDGJHGLlJO+SxEZwXqMmeK7iMkKf+Bxi6puBCIzzyiRtDgqvSSpuHVA5HbESKQkcb9nQy38gcfNLzb5LkJkDK4t7yIUcdz6rot8lyEyBvMw5mLfRUxGuAOPm8pa6bsMkTFK4a6f0GikDExlhb7XLHXlaoxp813ERIU38LiprBt8lyEyTnOBUPeSpGLWAaFfFyF1JY7rtIUyO4Sy6LJrAN1ZJGF0FcZM912EeKSpLAmvmcBlvouYiHAGHmPmAzrQTcIq1L0kmSR3QKqmsiTMrsCY0J15F74GV1NZEg0zgMt9FyFeaCpLwi5GCDttYbzReR2gnS4CQDFGqa+ZQqaFUm8Lpb5m6G+EYty92PLT0VhI9kNDFlLuT5Puw0w5SaKxz9vz4FKM2YK1GU9fX2rNXWy8xncZIhUwDVgLvOy7kLEKV+AxZiZwoe8yxI9ijNLJGeROzKbUORO620jm0iRxu58mLF6g2NxDfsopSm0noP0wiSndk/ucY5QArgIercHXEt/c7rzrfJchUkGXY8xWrM35LmQsjLXWdw1jZ8wdgO4lqiO9LeQ7FpM7vIB4z1QaiNXmgMlUlvyMI+TnHMDMOkgqmSdepS9lgR9gbWeVPr8EhTErcff9iUTJy1j7lO8ixiI8gceYC4DbfZch1dczldz+peQPLyCRaaHBdz2UsG2dZBftwM7bSzpeqvjatw6sva/Cn1OCxK11eD+ajpfoKQL/grWnfRcymjAFnnfjtsNJBFmwBxbTt2sVse7ppH3XM5JYgeK8/fQve63i014/wdr9Ffx8EiTGrCVCt06LDLEDax/2XcRowhF4jFkKvN13GVJ5hQSlXavo27OSVHk9TmhMP0rfRZuItXVWZBSqEze1FYInpIyLO1H5VyC4QV6kAv4Na4/7LuJ8gh943EK/XwZCe5y1nMuC3bWavtfXkCqkQrZ4fojpR8rBp2vSwecxrN1akaIkOIx5E/Am32WIVNlBrP2x7yLOJwyBZzU6dydSDi6g77UriGebarITqmbaD5O59ClSk9jmngH+GWsLlaxLPDImjRvdCdXopcgE3Y+1+3wXMZJgBx5j4riFfrpCIgJOTyH3wnpKQV6jM1mxIqUVr5Jd/hqNhgntKHsWazdVvDDxw5j1uLNKROpBF/D9oE7NB/2UxDUo7ETCjgvJPHoriSiHHYBSnNi2S2l69DZyp9qYyNkUa8pBX8LOmBZ0bpjUl2nAIt9FjCS4gcc1+pf6LkMmp6+RwuO3kN16GU02HuDHW4WdbqXh8Q0kt15C7zg/tAlYUY2apOYuhaqd3yQSVJf4LmAkQf4FtAxo9F2ETNyBhfT9/A7MqRnRHtUZUQyzYw3NT95EXz5JcRwfGdgGQ8bI3fm30ncZIh7MKd+KEDhBDjya9w6xrZfQu+nNNJYS6uF2zqbxkdspjWOKqw1jFla1KKm2VWihstSvQHbaghl4jJkDhO7qeYGSwT57PZkda7T2arD+RpJP3EKiYzFjvSj04qoWJNXjjtLQBaFSz5aU17AFSjADj0Z3Qqm/geLj76D/yHyafNcSRDZO7MVraNy5ekyh5wKMaa16UVINC4GpvosQ8SgGrPZdxFDBCzzGNAKLfZch49PfQPGJWyj2tNXpep2xMpgtl9M0xsXMF1W9HqkG7cwSgdXlO+QCI1DFlK0imHXJCPrTFB7fQLGvJVoHCVbTjjU0b7581NCzCmNCfQp13TGmGVjguwyRAGgiYIMXQQwWq3wXIGOXS1HceDPFbLPCznjtXk3ztrXnDT0pYHmt6pGKWMXEDpwUiaJAjXYGq/dozDxA6xZCohCn9OTbKWRaKnJ5Zl16/WKam0+Tmb9nxHVPywDdrxUGbrFy4NYtSP3pmUquu41iIYktJKGQxJ18bCFWwqT6obWT+JRTJOOlqg58XIAxLVh7uopfY8yCFXg0uhMqz19H9nSrFihP1kvrSDf2kp1xbNj1T3MxpgFr+2temIzXBUDgdqZItPU3UOycSe7ELEon24l1t5IqJcY44l7CprPkWropTO3CtnYSaz9CqqG/oseJLAZereDnm7DgBB63uCmwR1LL2batpffYPG09rwQbI/bMjSRvuJ988+lzzm4ZeF5s91CajM9i3wVIfcg0kd+1mtyhBST73SXMEzukN4bJNpHKNpE6Pqf8uhJ22gn65u/GXrCXdKIw6RGgJQQk8ATn8lA3nXWH7zJkdEfm0ffsDaQxWqtQSY2nyb3lXhLDDDHvxdoHvBQlY2fMB9AIj1TR4Qvo27EGTk6vTfsbK1Kau5/s8s0kpnRPeJ2mBb6LtdlK1jYRwRnh0ehOKPQ2k3/hzaQUdiqvr4XUK1eTueypc6YJ52NMEmvzXgqT0RkzA4UdqQILdv9SsjvWEMu01Pa6pVKc2IHFNB1YDG3H6bvsKeItPeMOPgb3+31b5SscnyDt0lLgCYHnr6NY1HURVdOxhKZD8+kb8uo47jA7CS61X1JxPVPJPXI7uZfX0eh7c8jJdhofvY3Ea5fRWzKMd2poSVWKGqdgBB5j2tDJpIG3czWZ7uk6WLDaXlpHsj9NYcirA9FgyIgUeKSitq+h97FbSfRODc4uWBsjtutCmh++k/zxWYxniuoCjPF+t1wwAo8ai8DLNJHfdonCTi0UUiSeX8/Q6auFGKORtSAypgkI5O3QEj6ZJvKPbSC7/RKabSwwv6PPkm0i9dTbSD/3ZjK5FMUxfEicABzIGZQfpgJPwL1wHYVSPDCPl8jrnE3j0bln9aASBKDBkGGp/ZKK2L2CzCO3EwvLSPrhhTQ9cjul7lZyY3h376PU/n+BGZMGZvsuQ0Z2YBF9J2fUdrGcwCtXErOcNVeuX6zBpP8XmbQX19G7+UqaSiFbI5lLk3ziFuLH5ow6xbXA991a/gOP67Vqx09AWbBbLgvXEzAq+lpI7Vp91gJmdQyCxt11Ns93GRJeFuzz68l0LA3vuWalBPGnb6ThwKJzNlwMlgJm1aqm4QQl8EhA7VlBX7ZJ92T5sn0tDYPmyNswJjALGAWAuQTreA8Jmeeup+/QogicWB/DbLqG9IGF5w09Xte6BSHwaLFfQBVjlF5fe87Jv1JDxSTx7WvPGirW8yVY9P8hE/biOnqPzI9A2BkQw2y69ryhp45HeIxJoctCA2vXarK5tAKPb/uXkS4kKJX/qWmtYGn3XYCE09ZL6A3zNNaIYpgXr6GhZ+qwC5nrOPDADM9fX0Zgwe5epbATBMUE8V2r3ugxeW0w5BwKPDJuR+eS3bEmgmGnzMaJPXsDdpgDCqeUNyp54TvwaDg4oA4u1OhOkOxZSaq8Y0uBJyhcw63rJGRcCnFKL14T/Y0gmSk0vHolmWHe5K0N8x141DsKqN2rfVcgg+XSJA8uJAs0lE8mF//Ufsm4vXJV/XQm9y2nech5YqDAI0HS20xe5+4Ez75lb/xVozzBoBFqGZfjs8geWBKhRcpj8MJ6Ev0NZ53GXIeBx92roQXLAbRn5ZhOzZQa65xJQ3nxsgJPMKjDJmNWiFPadK33QYaaK6RIbFpP/6BXeeso+Pzhz0AHDgbSkfn1MdwaNjZOrDytpZGFYFDgkTHb/Cb6+uv0TLPjc2jqmvFG6GnAGC+DHT4DjxqLAMo0kc+01OeTMgw6lmCAKb7rqHvuAEj9P8iYZBsp7K+zqayhtlz2xtEaAF7WIfoe4ZGAObTwnFu6JUA620nnkyR1c7p3ar9kzHZcSD+x+p7R6JxF48npb4zyeNmS7zPwRPYMgjA7tKC+n5SBF8Mcn00ObYf2ra576zJ2hTil/UvDcft5tb2+5o3Fy17aL5+BRw1GwBTilE5NR3c1BdzxOZTQdIpvar9kTDqWkC0mo3/uzlgcnUe6vGNLgUf8OjWdnI3V3y6CsOmcSRyN8PimEWoZk33L1KYOsDFie1aSpa6mtIyJgYb4gqZz5lmLyiSgTk8llU+qw+CZfv4yqkwT+e7p+l032KEFJKirwKPGIpBOahlmKNgYsYMLdTCkZ2rDZFT7l+pMs6FOTyFViPtpvxR45A3dbSR81yBj0z1NzyHP9POXUZ1s1yaQc8QwnbMoYkzNQ48CjwBQjFHqa9KBg2GRbdQaEs/UhsmoulvViRzOsTkU8TCtpcAjAPQ1U6j3cyLCJNeg55A3xiRAnQM5v3ySYr2erDyaE7P8bLxQ4BEA+pq0YDlM+htJlhf/S+2p/ZJRnZyuQ1xH0tNKqpCo/WJuXw2mFlwGTNZD4OkvECuUzowq9eWJl2ytqxherojpzY18dsZfP836knU1D/4eaiXbSBJ0tocnCjwyqq72s24IP696awttnNiOi2qfP3zNL0Z2XjMLph9irQz/YP8wrP8mPJmBeBpKSQjEw7qvqTZ1/Mbd/F4qTvZ4hsXXLeRHW45z6fEMS2c2saNoSXz8ar739Wd533sv5N6N+7l04OP+8DrubUhQ+vhP+M3Pv51vd/eT3N7JtIRxdT+4i8uLlsQ7lvEsQMFi5k+hZ8k0TgP878fY0Jame8k0Dl99AQc//VM+8dVb+eLUBgrD1fmXT3JzwlD8zHU8NNzbXzzCuq89g+nK0rb/FMtNuY58kYZv/xJ/Wemf21ClOLHNlxNfg3qRHihojiKs7WAlnZxx/l/+9d4WnphVP4HH61qR3ZB+FqYNPMnugsvzkPgo7gGSB7MKei6F0++GDbOg+1I4/E44eC184kX4YjvDPzg+ADcnoPivDP/geAjWfQTMEWh7DZabcg390NBB9X9RjiRboz7ru1dz9zMHuHhJG9s+eQ1PAE/81j18/Bt38LcD7/NfruRfd3TS3pun+abFPH3Pdm7ZdJiZ927n2pih2Jik+Honrc8dZHms/OTqytJuLbFnDrIcwFpMaQ67lkzj9KksyS3Hueor7+Dzn3yAT185jy/EY+QHP8FfOcL0Lz7Fhwb+XSiRLpRIfuhuLht43Tdu50sNCTcS9q5V3JMvEf/ddWwc/P195B4+VrUf3hBH55FYU6svJoNFYipR7WB1ZVrOH4zrvS3sT9c+f/gKPF4bjN3QdC8sj5efZIeg3ULsHtwDpASmCLvmQv9GuOpF+Pyb4NO3wxeSkB/8JH8Epv86Zx4c/ZAuQHIhZx4cW+FLTbgHxyfhnn6If4uzHxwrqN0vyuHkUrUJoStmcPxfNrP2m3fwZ0/sY+7fPseHpzdy8BP388Hufto/dwPfmN1MJhnnyEtH6PvBFu5sTnEqnaAQj53pLV4ym87+Ils//wSfbmngaLbAVIBT/bT35pjxy2v4+xsXcwDgS0/xjmScvgM9tJzOMeszP+NjJ7PM/9Dd/DfAfOoa7srkSbak6PrrW7nre6+w9lfW8mrJYp49wOxrF3D4Q3fz6YGey4d/xCe+fjtf/R+P8J77Xuc2gJYkXV+5lW/X4mc4oHdqdEdKAy4qgUftYBUV4+dvU+u9LexP136ktC4Dz03Q2Qdb3wefng5He2GqBY5B+0mY8Yfw978CB26Fd6ahbyu0dMKs6+FjR2H+QvhvFsx34K5uSE6Drpfgrj+GtZ+DV0tgfgyz3w2HF8KnY+UGZSl84jX46q3wnr/FPTjaoGsTtf1FORxbgx1a33+NVQ/t5q3NKbp+/0E+fNkcnp8/lW3zp9CxZhb7vvsy79vRyfR/3sx7b1vBfQBNSbpH+nzpOMVZLexY2MquA90sKlnMglb2HOhmYXPS3cr7k9dZsvcUKxMxcv9vMzfctoLv/tYVPP+bP+JT37qTL+WKmEQMmy8Sa4hz9yd+wgcxcPNSXu/JkfrOS7znuy9T/MgV/EMq7v4fY4ZiKo49nmHut+7kS1DbkZ0BfSYav3hDKBK7GdUOVlfpPIFHbSEUk/UzwuO9wWiG4mLYsQZ2bYNFJTAXwp5tsLAV+r8BSzbDyhTk/gxu+K/w3S/B80vgU7vhS1kwSbD9EGuCuy+HDxrgN+D1Tkj9Abzns1D8CvxDuvxEj0MxDbYD5u7GPTiC0qOxpvpz6HeuYvsvrWb7f72X35k7lX3TG+nZ2QUHTzP3ktnsBbh5GftuXMyXt5+g7cXDrLYQMyOsA3hiHysyeaZuPc5lA72a3jxtAC8cYtVtK9gdM5Q+dQ3/92vP8IEPXs5D86fQO/hzDDxxGxKUvvcKG66cxwv/6VJeBpjVTP837uDr/7aFlfe9zjXXLuCHQ2v4zR/xKYBcsfYL8ft175kvkfm5qx2sHnue33JqCyHvocPmK/B4X6D2r7CiG6b+Ai4b6NmcxD1AHoBVd8CWb8P//Sh84PPw0CrOfnDdvJnLAAAYu0lEQVQMPHmboPTHsOE2eOF/4x4ci6B/G3z9L2Hl1+Gad3Pug2MJ7sGRDciONVOD/5GBJxTAH93AT57uYNazB+F0jtal0+iiHITv38GyY71M7cvTFDPuCZ7JkyzaM0+Qbcdp7cnRfPEst95g3ymWWItZ1MauTJ6mnV2svX8HW9+xnD0DH/N3z7Ph8GmWGENpYBjXWuJXzGXj765j41/dwj++//v8yUO72VCy7rkRMxSLlsR3382fDvc9+RzhaSj5fx7Vqcj83NUO+qG2EGK29s+jugw8T0NrFzS/pbw4bzMsKYG5GHZ1Q9MLsPY22HoTdAJ8EjbshCUxKA0M5ZYgvgE2fgs2/gL+cTr8yXdgQ7H8M41DsQCJwwz/4Ahaz8bY2o+6tTSQW9LGnmcP8ub2JvrXzOQ5gJ/v4dp3ruTnB08zu72RzgM9zFs5g65knCf/4WVuB1jVzqnm3WSeO8h1gB3o1XTnmFGyJKalOTz4CQ7w52/n+wN/HxjGHVrTP/8HPgfwlae5DuAT63hipPqLlsRAryZbYMpkfx7j1eihwRAgIoFH7WB1xcbRIanHtjBRGvu2/UrxFXi8HnK3Dk61QuY+3ANkoGdzAmYUITEHDn+UMw+QRznz4BgYyh36OTtxD47fcJ+Tuxj5wVGAxEDP5jS1/0U5nFixNo14yYItTwmsmcnJx/Zg5k9lx6ce4KNXzeOpnV1MOZlldr5IfHcXF/739Xz580+y4b7XWfWmeezL5N2TGeC3ruDp376Sp471kr7rRa6fmuL071zNL4720vDADlYN/rp2lGmIQgnTmyMxpYF8bJjo15d3C+wak2eepN+6ky8P873VLDgmcsPvkJGqi8QhnWoHqys2yq/zem8LU/31E3i895C+DE//DTy1D9K/D9fPgNPfhF/shYb/w5kHSGmUB0ceTBckZkB+uCXnPeUzO6YMmnvdzdkPjiK1/UU5nHRfbf5PPvhD/mDZNF7pyxP/0ye443iGuX91M393Okfy9x/kU/Om0HXrcn7cmib7O1fx3YYEpT+6gZ8A/HQniy6d7XqjcGZY+IVDzN13imUfvpwfgJtvHph7BvfkGzqvXCiR6s0Rb065/5etx2n7/EY+Go9RYNDj80N3sw7AWuLvXMXd772QbUVLsi9PfPATvmTh1/+dzy5uY3NVfnDDaOtU4PEkEoEH1A5WUzJ3/sdJvbeFqWzt2y9jrYfsYcxbgJW1/8Ln+gYs+Qps+CL84FY4NvhtRWAhfOYA/MXA6y6Az7wGfzVwoNZjMO398NEEZz84BpQg/rtw93+HbUvgUy/DVwc/6YvAPPjsxbD5Z8PMcdfK7pVkNr+ptifI/nQni25awr5EzP3cTmRIzWgiV8saKul4hob2Jrcjouos9upH+c6sgza0P6/QMmYecIfvMipJ7WDlvXwVmX3Lx9am1mNbOLuDjVc9ZmvWQQR/geda4OLaf+Fg2g8NC6jRL8oRHJpP3/PX19fCwTBLZcnf8u98F2s1ylNrxrQD7/FdRtQEoR2spP1LyLx0ja4hGcm1P+OxGUft1lp+TV/bKzOevm4gBeFJ3piJzlbbepDuo6Cw443aryoIQjtYSdOO62DQkTT0kZ9xrPZreBR4BIB0n+4HCpNUlqzvGupYHwFYhyjB1tJDKlaMznqvSmo7QZ4hRxzUggKPAJDuIxHP1z5xy8Q09Ne+sZAytw6gz3cZEnzN3brcdzjthwEFHvFpSnd4F8nVm3RGzyHP9POXUbV2aSflcGYeJoECj/jU2qnh17CY06HnkGf6+cuoZh/Q2sihEjkKLT2UfKxB9POfYW0/I9wJIv5MOx6dMzCiLJUlP+2EplQ8U+CRUc0+QFoHhJ5t2nFyeBjdAb+X4KnBCJhpx0j6rkFG19rpZ8GfnEXtl4wqZjFzOqK1+2yylm4jBpz28bUVeOQNzb0kG/q0yC7o2o8AnhoMeYPaLxmTJdvUkRyQzpCbeZg0GuGRIJh5WIEn6Nrdgj8FHr/UfsmYtJ4k1dqpYyQAFm9/Y3qv7kZ4Tnn82jKCeXu1yC7IGvrIt54kjrX6heuX2i8Zs+WbdW5TIkdh8XbS5X/W3QjPcY9fW0bQfpgGHZYVXHM6/C34k7OcBC1GlbGZ20Fjure+j/1Yso3+RPGNzNHjowYFHjlLzGKmH9Pwa1At2EkC98tWfHKHD3b6LkPC47KnKGHrc6Qnnqe4bMsbdzVa4ISPOvwFHmu7ob4Tb1At2Knt6UGUzpBr66IBOOK7FgGG3Coucj7tR0nP31Ofx0lc/NxZozsnsdbLWlHf6zU0yhNAc/fr7IggmrP/jQXlR70WIgPUfsm4rH2WdL3thJ3dQd/8PWfdGu+t/VLgkXPELOaCvTo7IlBK2KVbSeGGgxV4gkHtl4xLokjssqfq59Ddhj7yl/+ChiGvVuCRYFn2Gql6nW8Oovaj9DVlSAJdvoaD5Rxd6MR4GaeZh0nP21MHxxqUsFc+TilROCdnKPBIsDRlSLYfqc/55iBa+Qrx8l81uhMU1pbwtPhSwu2SZ0k3ZKK9hnX5FjLTTpwzulPAdRS88Bt4rD0J9TWfGSYXbiLhuwaBqZ1kpx9/o+FQ4AkWddpk3BIFYut/honqWsnWE2RXvXzWup0Bx8sdBS98j/CAekiB1XqS1MyDdTD0GnCrXz7rnwo8waLAIxPS3Evy2ocoxQvRmhad2kX22odIGYbd7eu1/QpC4DnkuwAZ2drnSVLSWh5fph0jO+vQG6eT5vA4HCzDOuy7AAmv1pOkrnmYQlRCz5STZNc/SGrQFvShvB7lEITAs9d3ATKy5tMk5+7XWh4vSthLnjnrOXqsfOCdBIWbltc1EzJh007QsP5BimGf3mo7Tt+bf3resAOezxDzH3isPQr6hRpkazaRikWkBxIm8/eQmdJNatCrDngrRs5HnTaZlNaTpN78IKWwLmSes4/M+p+RHiXsnMBar5ce+w88zj7fBcjI0n0kVr+sc3lqKZ6nuOaFN6ayBuz2UoyMRu2XTNqUblJv/TGJ+bvDs27SlCiteJXeKzfSFLOjntC/pxY1nU9QAo96SAG3dBtNrSd0x1atXLSJ/mT+ja3o4M7f0dRJMB0GdQhk8soHEzZd/QjZVDbYO5indpK98T4Kq16heYwf4r3DFpTA04EO8Aq8KzYSN7pJvepmHiSzaOc5Wzq9NxYyArfNdr/vMiQ6Zh0i/dYfE5vdEbzRnnie4tpnydzwAOmWnrOm3M+nG2u9X7YbjMBjbQE46LsMOb/mXpKrXtEoTzWlsuSvePKcw7pAgSfoNEotFZXME7/qcZou30hfOiBre2Z3kLnpHli8Y9gzds5nTzXqGa8gHSy3F1jguwg5v+VbaDo+h8zxOeN+wMtoLPaKjRSTeZJD3tKNtTqvKtj2AyWC0omUyLhgH43z9mEPLqJv+1pivVOH7RBVTwnbfoTsis2YGccm3O4HosMWtMBzne8iZHRXPkb60dvI9bWMeThTxmDpVjLtR4edDw9EYyHnYW0OYw4D83yXItFjwFywl8YL9sKJmWR3r6J0ZB5pG69ewG7oIz9vL7llW2lI99E4iU+VwVqv29EHBCfwWNuLMceBdt+lyPklisSueRjz2G0Ui4mzFtbKBM3uIHPRiyMu/lPgCYe9KPBIlc04RnrGMcgnKR5aQObEbDg5nXhvCylio+6UGlkJ25ghP+MYhYU7iJevsxk62jwRgZnuNYE6x8yYS4BrfJchY3N0LtlnbqQBM4knmTC1k+x1P6VhhG2dvVj7vZoXJeNnTBPwqwx/pL5IVRVjlLrayZ+YRaFrJibTQrwYJ15IYErx8mPSgrHYRIFSSzeFqScptXZiWjtJtHSTHMPW8om4D2s7qvB5xy04IzzOduBqNA8eCrMOkV77HJlXr9J6nolK95K79uHzNjQa3QkLazMYsw9Y5LsUqT/xErH2ozS0Hx3TGp84lRm9GU2WAG1IClawsDZLQFZzy9gs3kHT6heDt3UyDFJZ8ut/hhly3s5Q22pWkFTCVt8FiATINp+3ow8VrMDjqMEImeVbaFr1kkLPeKSy5K97AJoy5+1lHdHurNDZB/T6LkIkILb4LmCw4AUeN9fX47sMGZ8Vr9F04SYyWN2sPpp0htz192NGCTsAm2tSkFSOWxSpUTkR6MDabt9FDBa8wOO85rsAGb9lW2m6YiPZmE5jHlHLKfqvv594Y9+o6+eywK5a1CQVtwX0HJC6F7jf40ENPFuBgu8iZPzm7adx/YPkg34PjA8zD5K57gGSDf1j2sq/JUhz3zIO1vaitYhS33oI0Hb0AcEMPNb2A6/7LkMmpq2Lhht/gmk5pQsVAShhV71EZt2jNCWKY3rOldB0Vti96rsAEY9eIVBn3jjBDDyOGowQa8iSuP5+kgt20lvP63qS/RTWP0RuxWvj2rq/A2u1CDzMrD0MHPddhogHOQK6ji24gcfaLnQDcajFS8QufYbmdT+nvx6nuNoP0/eWezHlE0vH46WqFCS19rLvAkQ82IK1gWzvgxt4nGd9FyCTN/MI6ZvuIT5nX31sXU9lyV/xBH3X/JzGMa7XGWx/OexL+O0EOn0XIVJDJQI8OxOsqyWGY8zbgaW+y5DKODaH7CtXYjJTanzjby1Y7Pw99K19jnSiMOHOxI/K0yESBcYsAjb4LkOkRl7F2id9FzGSMASeVuCXCf5olIzD3mVktl1CMpeuyfHmVTf9CH0XbSLW1jWpILcbax+sWFESDMbcCczxXYZIleWAfy7fmBBIwQ88AMZcD1zouwyprGKM0o419O1eSUMhFbh73cZk+tFy0Omc9IhVCfjXoB3UJRVgzBzgTt9liFTZM1j7ou8izicsgacZeB/Bu+xUKqAYo7RvOdmdq0lkm0n5rmdUJWz7UfpWv0S8AkFnQKCHgmWSjHkHsNB3GSJVchr4F6wt+i7kfMIReACMWQdc6rsMqa5js8nuupDS8VmkbTxY05gNGXILdpNfsp2GhmxFw3cO+Kfy+VMSRcZMB94LGN+liFTBI1i73XcRownTiMmLuGmt4I8AyITNPEJ65hEoxCkdmU/fgUVwfDapUmLcu50qovE0ufYj5BfsIlHeXl6Nx98LCjsRZ20nxuwElvsuRaTCThCSg4LDM8IDYMzlwFW+y5DaKhnssblkj82l1DWDeE8ryWoFoESOQlsnudkdMKeD1BjuvJqsHtzanUAPBUsFGDMFNzUfqJFLkUm6r3zpd+CFLfAkgP8ItPguRfyxYHtayZ+YRaGnFfqaMdkmYv1pYrkGEpjzTxvECxQTOYqpHKXmbkqtXdDWSWxq15jvuaqkh7B2Z42/pvhizLXAxb7LEKmQDqy9z3cRYxWuwANgzHzgNt9lSDBZsLk0xWIMW0xgbTn6GAupHLFUlrgJzjqKo1j7Q99FSA25TtsvA1N8lyIySRb4N6w94buQsQrTGh7H2g6M2Qas8l2KBI8BU+EFxdVSAp7wXYTUmLUFjHkUuMN3KSKT9FKYwg6Edy75F0Cv7yJEJuEFrNXlkvXI2oPAa77LEJmEE8BzvosYr3AGHmtzwGO+yxCZoKO4XYdSv57GLVgXCZsS8HOsLfkuZLzCGXgArN0PBH7fv8gQBULaWEgFudukH/VdhsgEPIe1obwUN7yBx3kS6uMGbomMp7H2lO8iJADc1NYW32WIjMMR4CXfRUxUuAOPprYkXDqwdrPvIiRQnsIdyy8SdAOj0yHb2n1GuAMPgLX70NSWBF8/8IjvIiRgNLUl4fFU2C83Dn/gcTYCXb6LEDmPjVir6Vc5l7UH0CJ2CbYOrA39zsJoBB7XS3oA14sWCZodWLvDdxESaM8C+3wXITKMDBEZnY5G4AHKQ20P4U5/FAmKo2idmYzGrYt4GDjpuxSRQQrA/VEZnY5O4AHKF5g97bsMkbLTwE+xtuC7EAkBtwnjASDnuxSRskeidEBqtAIPgLUvE5Kr6iXS8kSoZyQ14o4s0Ei1BMFzWLvLdxGVFL3A4zwGHPNdhNQti7sFPZSHc4ln7lBVjVSLTzuw9gXfRVRaNAOPtUXgp+hQQvHjqfJxCSITo5Fq8ecoET0qIZqBB8DaXuBBoOi7FKkrr2HtK76LkEjQSLXU2mnggfKgQeREN/AAWHsENx+ue4ukFjpw152ITJ77pXM/2rkltTGw7rDPdyHVEu3AA2DtHuDnaBGgVNcJ4Ge6FFQqyv3yuRcI9Qm3EngF3I7SSK87jH7gAbB2JzoLRarnBHBveVuxSGW56fl70Z1bUh1uzas78TvS6iPwAFi7DXjCdxkSOQNhJ+u7EIkwa3twoUcbMaSSirg1Ox2+C6mF+gk8QPkuEIUeqRSFHakdd0bPPUCv71IkEorAg/USdgBMiG96nzhjVgE3AMZ3KRJax4D7sFb3t0ltGTMVuANo8V2KhFYBN7IT+Wmsweoz8AAYswJ4Cwo9Mn4HcY1F3nchUqeMacGFnqm+S5HQyeF2Yx32XUit1W/gATBmMXATkPBbiITIPtwwcCTPqZAQMaYZ2AC0+y5FQiOLG5mOzP1Y41HfgQfAmBm4RkPDwzKa14FHtfVcAsOYBG6keqnnSiT4TuFGpuv2XCcFHgBjGoFbgNm+S5FAsrjrInSCsgSTMVcCV/guQwJrH/BwvR+docAzwJg4cD2w0ncpEihZ3IGCB30XInJexizFjfZoil4G24S7+bzuf9kr8AxlzKXA1Wgxs8Bx3IFcOvBNwsGYdtwUfbPvUsS7AvAI1u7yXUhQKPAMx5hFuMXMSd+liDfbgce1OFlCx5gm3BT9LN+liDfd1MFVEeOlwDMSY6bjGg1t+6wvJdx6nVd9FyIyYZqir2cdwEM6I+xcCjzn43ZAXANc5LsUqYk+3HqdQ74LEakIY5YDbwYafJciVWeBl4FntF5neAo8Y2HMBcCNaOt6lO0ENuqaCIkcN8V1PbDIdylSNSeBx+rxMMHxUOAZK2NSwLXAKt+lSEVlcGt19vouRKSqjFkJrAdSvkuRirHAS8DzWm84OgWe8TJmIe4eribfpcikbcWt16nrsymkjrjTmW8AFvguRSatE3cQ6jHfhYSFAs9EGNOAmxdf7rsUmZBu3PCvztaR+mTMatyItXaihk8Jd7bOJp36Pj4KPJPh7uK6FpjitxAZIwu8gjuEq+C7GBGv3AWk1wELfZciY3YMN6qj7eYToMAzWcbEcLu4rgDSnquRkR0FnsTao74LEQkUY+YC69C5PUHWB7wAvKYdWBOnwFMpblHzpcDF6Gj3IDkBPIu1+3wXIhJo7mqKq4BW36XIG/pxi5Jf1aj05CnwVJrbAnolbjeXrqfw5yRu6krHqouMlRuxXo0bsdbGDH8KuOn3l7SponIUeKrFmGm4O7l09kVt9QDPA69r6Fdkgtyhq5fgRq21sLl2isAW3ILkPt/FRI0CT7UZMwe4DC0MrLZe3Bz3Nu1cEKkQY9K40HMhOr+nmizu/r7ndVlx9Sjw1Ioxrbj1PSvRGp9K6gQ2A9t18JZIlRiTxLVdF6P7BSupH9gGbMbaHt/FRJ0CT625M3xW43pMajgmpgTsxjUSOkpdpFaMMbjR6jXAfM/VhNkJXEdthxYj144Cj0/GzMcFn0VAzHM1YXASN+y7HWszvosRqWvGTMG1X6uARs/VhEEed2ffVh2P4YcCTxC4nV3LgcXAbLS7a7CBRmIb1h7xXYyIDOF2di0CluKurNBan7Mdxl1js0ujOX4p8ASNWyS4CBd+5gNxr/X40QPsA/YDB9VIiISECz/zcO3XIqDZaz1+5IEOXBu2T7utgkOBJ8jc1tAFuMZjIdDgtZ7qKeF6QQMNxEnP9YhIJRgzkzMduOl+i6mqUwy0X3BIO0WDSYEnLFzPaQ6u9zSz/BLmqyx6OdML6sDavOd6RKSajJmK67jNwrVfYT7ROYe7rmY/rpN2ynM9MgYKPGHmFg3O5EwD0k7wDgmzuMXGJwa9HMfarNeqRMQvdx3PzCEvLV5rGl4GOM5A2wUnsLbbb0kyEQo8UeK2jLZxpvfUjGtAmsp/VvP8nxzugrsM0MWZBqJT5+OIyJi4NYwDnbcpuLarufxSzRHtAq796gNOMzjgaA1OZCjw1BPXo2oe8pLGLYyO47bGD/xpyy+U/yxxpkE490WhRkSqyZg4ZwLQ4CCUxLVZg9uvAQNraSxuMXGGM+3Wmb/rvqq6oMAjIiIikafD7kRERCTyFHhEREQk8hR4REREJPIUeERERCTyFHhEREQk8hR4REREJPIUeERERCTyFHhEREQk8hR4REREJPIUeERERCTyFHhEREQk8hR4REREJPIUeERERCTyFHhEREQk8hR4REREJPIUeERERCTyFHhEREQk8hR4REREJPIUeERERCTyFHhEREQk8hR4REREJPIUeERERCTyFHhEREQk8hR4REREJPIUeERERCTyFHhEREQk8hR4REREJPIUeERERCTy/n/dS7AtOeLqbwAAAABJRU5ErkJggg==)

### 总结

我认为这是一个比MSE更有效的针对多分类问题的评判标准。

p-score的优势在于，它能直观的表现模型的预测能力。我们能够明确的说p-score=2的模型比p-score=1的模型预测效果好$e^{2-1}=e$倍(自然对数)，但却不能通过MSE，召回率或准确率来量化的比较模型的预测能力，也就是没有办法说清楚MSE=0.01与MSE=0.04的模型相比预测效果到底优了多少。而召回率，准确率也是这样，你能说100%准确率的模型比50%准确率的具体好了多少吗？

所以我认为p-score更符合概率论，或数据统计中的表达方式。

同时，它也能避免最初提到的赛制漏洞，我没有想到能根据p-score来调整模型的方法，而诸如MSE，召回率，准确率，我认为都是利用得分参与模型调优的。（MSE的利用方法已提到，而召回率，准确率我猜测能通过回归预测各样本的分类置信度和得分来调整结果）MSE等判分方法，都只防君子，不防小人。禁止通过考试答案来准备考试是有效的，但禁止通过以前的考试分数来准备考试是难以操作和实现的。

p-score也存在缺陷：
 - 没有考虑分类结果的逻辑相邻：分类结果数值上的相邻导致了逻辑上的相邻，但这是没有被考虑到的。（例如：把幸福等级为"很幸福"的样本分为"比较幸福"与"不幸福"都是错误答案，但错误程度是不同的。）
 - 计算复杂度高，未经优化的代码在上面的测试中（1000条数据）每次运行需要39.4 ms(± 72.8 µs),平均运行时间是比MSE方法的394倍，是accuracy方法的261倍。但可能在优化后会提升效率，展示用代码实用性较低。
 - 一点也不优美:需要想办法优化

参考文献：6 Mathematics of a Lady Tasting Tea By SIR RONALD A. FISHER Uddrag af James R. Newman, The World of Mathematics, Volume III., Part VIII, Statistics and the Design of Experiments (New York, Simon & Schuster, 1956), pp. 1514-1521
>这其实是我概率论课程的作业，由于时间限制，我没有查阅足够的文献，这个notebook仅为我阅读费舍尔著作的论文时顺手实现的一点想法。
