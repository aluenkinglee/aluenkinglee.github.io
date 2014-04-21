---
layout: post
title: "a scratch of feature selection in traffic classification"
date: 2014-04-20 21:02
comments: true
categories: 特征选择 机器学习
---

特征选取在减轻识别流量监测方面起着很重要的作用。该方法可以显著提高计算流量分类的性能。但是，大部分的特征不能应用在实时在线的流量分类中（有些特征只能在获取完整个流量才能得到，比如传输的数据大小，流的传输时长等）。所以在抉择分类的时候，需要一个优化过的特征集合在更短的时间内完成流量的分类。另外一种方案就是使用新型的网络架构如SDN/OpenFlow在目前已有的特征选择方法中，Chi-squared, Fuzzy-rough and Consistency-based的特征选择方法最适合P2P流量选择（那现在的手机端流量分析怎么样？）这些算法在使用ML进行在线P2P检测时会给出较好的特征子集。
<!--more-->
特征选取是寻找一个最小特征子集，可以快速有效的识别出实例的类别。如果利用一个特征进行分类聚类的结果与不使用它的结果没有很大的差别，则称整个特征时没有分类能力的。使用这些具备分类，聚类能力的特征，在分类的准确性和计算性能上都会得到提升。[1]主要研究的是在线流量分类中的流特征问题。然后考虑精度和性能的因素，选取了3中能够应用到P2P流量中的特征选择方法。

[1]实现的主要方法是使用了几个特征选择算法来提出在线的流量特征，使用J48算法作为分类器。

特征规模大小与分类器的效率和准确率息息相关，最优的特征集合可以减少分类器的建模和检测时间，从而提升分类器的性能[5]。主流的分类器有CSF, CON, Filter-Sub, Fuzzy-rough, Symmetrical-Uncert, Chi-squared,Info Gain, Relief, Principal and Latent-semantic。作者使用的Chi-squared, Consistency and fuzzy-rough算法，相关文献可以在论文[1]中找到。

## 在线特征提取

尽管Moore提出了248中流量特征，这些特征源自于同一个流中的报头信息。实际应用中的确不能全部都用到。具体操作是对现有的特征集使用那十个特征选择方法，分别选出各自的特征子集，然后应用到SVM分类器中，判别的准则为建模时间(训练时间)和准确率。然后合并准确率最高的前3个特征集合的并集作为最优特征子集。然后在分出那些特征可以在线获取（SOF-selection of features），他们作为分类器的输入-报文的特征向量。

核心思想是使用监督方法对有标记的数据集进行分类时，对特征集合进行规约，减少特征集的大小。

#### 实验


**信息增益**：特征 $A$ 对于训练数据集$D$的信息增益 $g\left( D,A \right)$,定义为集合 $D$ 的经验熵 $H\left( D \right)$ 与特征 $A$ 在给定条件下 $D$ 
的经验条件熵c之差。

$$
g\left( D,A \right) =H\left( D \right) -H\left( { D }|{ A } \right) 
$$

给定训练数据集$D$和特征$A$，经验熵$H(D)$表示对数据集$D$进行分类的不确定性，
而经验条件熵$H\left( { D }|{ A } \right)$表示在特征$A$给定条件下对数据集$D$分类的的不确定性。他们的差
就是`信息增益`。表示由于特征$A$而使得对数据集$D$的分类不确定性减少的程度。
显然，对于数据集$D$而言，信息增益依赖特征，不同的特征具有不同的信息增益，信息增益大的特征具有更强的分类能力。


所以算法选择特征的准则就是：对于训练数据集$D$，计算其每个特征的信息增益，并比较他们的大小，选在信息增益最大的特征。


设训练数据集为$$\left\vert  D  \right\vert $$表示样本大小，在我们这里就有42个实例。设有$K$个类$${C}_{k}$$。
令$$\left\vert {C}_{k}  \right\vert$$为属于类k的个数，即$$\sum_{k=1}^{K}{\left|{C}_{k} \right|}=\left|D \right|$$
设特征A有n个不同的取值， $$a_1, \ldots, a_n$$，根据特征$$A$$的取值将$$D$$划分为n个子集，$$D_1,D_2,\ldots,D_n$$,$$\left\vert D_i \right\vert$$
为$$D_i$$的样本个数，$$\sum_{i=1}^{n}\left\vert D_i \right\vert = \left\vert D \right\vert$$.记子集$$D_i$$中属于类$$C_k$$的样本的集合为$$D_ik$$,即
$$D_ik=D_i \cap C_k$$,$$|D_ik|$$为$$D_ik$$的样本个数。于是信息增益算法如下：

**信息增益算法**

输入：训练集D和特征A

输出：特征A对特征集D的信息增益$$g(D,A)$$

1.计算数据集$$D$$的经验熵$$H(D)$$


$$
H(D)=-\sum_{k=1}^{K}{\frac{\vert C_k \vert}{|D|}}\log_{2}{\frac{\vert C_k \vert}{|D|}}
$$


2.计算特征$$A$$对数据集$$D$$的经验条件熵H(D | A)

$$
H(D|A)=\sum_{i=1}^{n}{\frac{|D_i|}{|D|}}H(D_i)=-\sum_{i=1}^{n}{\frac{\vert D_i \vert}{|D|}} \sum_{k=1}^{K}{\frac{\vert D_{ik} \vert}{|D_i|}}\log_{2}{\frac{\vert D_{ik} \vert}{|D_i|}}
$$

3.计算信息增益

$$
g\left( D,A \right) =H\left( D \right) -H\left( { D }|{ A } \right) 
$$

######使用信息熵增益算法对特征进行降纬

实验前

```text
Instances:    42
Attributes:   35
              sip
              sport
              dip
              dport
              protocal
              interval1
              interval2
              interval3
              interval4
              interval5
              interval6
              interval7
              interval8
              interval9
              packet_len1
              packet_len2
              packet_len3
              packet_len4
              packet_len5
              packet_len6
              packet_len7
              packet_len8
              packet_len9
              packet_len10
              payload_len1
              payload_len2
              payload_len3
              payload_len4
              payload_len5
              payload_len6
              payload_len7
              payload_len8
              payload_len9
              payload_len10
              cluster
```

得到的属性的排序是

```text
Ranked attributes:
 1.944968503161257472   12 interval7
 1.79590506127720192    22 packet_len8
 1.789000161744105216    9 interval4
 1.485875203840541952   32 payload_len8
 1.476115354039271936   24 packet_len10
 1.474554148784290048   19 packet_len5
 1.433523944277332992   11 interval6
 1.4172094615274624     20 packet_len6
 1.40428856745235968     6 interval1
 1.326924790992714496   34 payload_len10
 1.181689898847241984   18 packet_len4
 1.105237515513706624   23 packet_len9
 1.009802257207393664   28 payload_len4
 1.000000000000000896   31 payload_len7
 1.000000000000000896   30 payload_len6
 0.993447238380203776   21 packet_len7
 0.868563607479333888    3 dip
 0.832352013234566144   14 interval9
 0.829607103088203904   29 payload_len5
 0.781988055469156096   17 packet_len3
 0.781988055469156096   16 packet_len2
 0.737113917996471168   10 interval5
 0.544053730963280448    2 sport
 0                      27 payload_len3
 0                       5 protocal
 0                       1 sip
 0                       4 dport
 0                      33 payload_len9
 0                      25 payload_len1
 0                      15 packet_len1
 0                      13 interval8
 0                       7 interval2
 0                      26 payload_len2
 0                       8 interval3

```

所以对于属性payload_len3，protocal，sip，dport，payload_len9，payload_len1，payload_len2，packet_len1，interval8，interval2，interval3这十一个属性都可以去掉。
降维之后的属性集合大小为23.

降维之后的聚类结果

降维之前的聚类结果

效果非常吻合。

###### 使用PCA对特征集规约

这个使用weka的来做的，做出来之后有12个特征（都是原有特征的线性组合）

```text
Attributes:   12
              0.333payload_len7+0.332packet_len7-0.307dip+0.295payload_len6+0.293packet_len6...
              0.399packet_len8+0.398payload_len8-0.269payload_len5-0.268packet_len5+0.261interval5...
              0.399interval4+0.309interval1-0.308packet_len10-0.274packet_len9-0.272payload_len9...
              -0.366payload_len9-0.366packet_len9-0.349packet_len5-0.347payload_len5-0.288interval1...
              -0.357interval3+0.351packet_len7+0.348payload_len7-0.324interval5+0.313interval7...
              0.454interval6-0.424interval5-0.401interval1+0.262interval9+0.256packet_len4...
              -0.402interval7-0.388payload_len6-0.387packet_len6-0.357interval8-0.26sport...
              -0.597interval9+0.443packet_len10-0.427packet_len2-0.239packet_len3-0.209packet_len6...
              0.865interval2+0.178interval3+0.168packet_len10+0.149packet_len4-0.146payload_len9...
              -0.751sport+0.346interval3+0.286dip+0.216interval7-0.178packet_len2...
              -0.434interval3+0.425interval2+0.315payload_len9+0.312packet_len9-0.311sport...
```

降维之后的聚类结果

分析，由于降维特征减少太多，走势已经不太吻合。

###### 卡方分布提取特征算法

对于这个算法，这里只给出特征选取的结果

```text
Ranked attributes:
157.0864   12 interval7
145.6184   22 packet_len8
130.1739   20 packet_len6
123.7833    6 interval1
123.0833    9 interval4
118.16     32 payload_len8
108.7528   34 payload_len10
108.5061   24 packet_len10
103.1333   18 packet_len4
 92.3818   11 interval6
 85.5423   23 packet_len9
 81.0409   19 packet_len5
 61.9733   28 payload_len4
 42        31 payload_len7
 42        29 payload_len5
 42        30 payload_len6
 42        21 packet_len7
 39.4135   16 packet_len2
 39.4135   17 packet_len3
 38.4      14 interval9
 37.9167    2 sport
 36.9542    3 dip
 33.9      10 interval5
  0         5 protocal
  0        33 payload_len9
  0         4 dport
  0         1 sip
  0        25 payload_len1
  0        15 packet_len1
  0        13 interval8
  0         8 interval3
  0         7 interval2
  0        26 payload_len2
  0        27 payload_len3
```

被取消的特征同样是那11个特征，只是排序结果不一样了。
所以理所当然kmeans实验室一致的。
Kmeans实验和实验1一样

最终选取的特征集合为23个

#### 参考

[1]. Jamil, H.A., et al., Selection of On-line Features for Peer-to-Peer Network Traffic Classification, in Recent Advances in Intelligent Informatics. 2014, Springer. p. 379-390.

[2]. Zhen, L. and L. Qiong, A new feature selection method for internet traffic classification using ml. Physics Procedia, 2012. 33: p. 1338-1345.

[3]. Moore, A.W. and D. Zuev. Internet traffic classification using bayesian analysis techniques. in ACM SIGMETRICS Performance Evaluation Review. 2005: ACM.

[4]. Dash, M. and P.W. Koot, Feature selection for clustering, in Encyclopedia of database systems. 2009, Springer. p. 1119-1125.

[5]. 统计学习方法。李航

[6]. Mitra, P., C.A. Murthy and S.K. Pal, Unsupervised feature selection using feature similarity. IEEE transactions on pattern analysis and machine intelligence, 2002. 24(3): p. 301-312.

