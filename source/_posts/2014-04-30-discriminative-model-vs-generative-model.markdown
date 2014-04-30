---
layout: post
title: "Discriminative Model Vs. Generative Model"
date: 2014-04-30 17:17
comments: true
categories: 机器学习 Machine learning 整理
---

## 判别模型和生成模型分析

在学习复习ML内容时，中文检索生成模型搜到了该[这里](http://blog.sciencenet.cn/home.php?mod=space&uid=248173&do=blog&id=227964)本文主要参考该文章，并稍作整理。

<!-- more -->

这两者进行预测的方式不同在于模型的处理上：

**生成模型**：无穷样本 ==> 概率密度模型 = 产生模型 ==> 预测

**判别模型**：有限样本 ==> 判别函数 = 预测模型 ==> 预测

简单的说，假设$$x$$是观察值，$$y$$是模型。

如果对先验概率$$P(x|y)$$建模，就是**生成模型（Generative modle）**。
其基本思想是首先建立样本的概率密度模型，再利用模型进行推理预测。要求已知样本无穷或尽可能的大。
这种方法一般建立在统计力学和bayes理论的基础之上。

如果对条件概率(后验概率)$$P(y|x)$$建模，就是**判别模型（Discrminative modle）**。基本思想是有限样本条件下建立判别函数，不考虑样本的产生模型，直接研究预测模型。代表性理论为统计学习理论。
这两种方法目前交叉较多。

### 判别模型Discriminative Model

又可以称为条件模型，或条件概率模型。估计的是条件概率分布(conditional distribution)，即$$ p(class|context)$$。按照上文的记法就是$$P(y|x)$$
利用正负例和分类标签，焦点在判别模型的边缘分布。目标函数直接对应于分类准确率。

######主要特点：

寻找不同类别之间的最优分类面，反映的是异类数据之间的差异。

######优点:

1.   分类边界更灵活，比使用纯概率方法或生产模型得到的更高级。

2.   能清晰的分辨出多类或某一类与其他类之间的差异特征

3.   在聚类、viewpoint changes, partial occlusion and scale variations中的效果较好

4.   适用于较多类别的识别

5.   判别模型的性能比生成模型要简单，比较容易学习

######缺点:

1. 不能反映训练数据本身的特性。能力有限，可以告诉你的是1还是2，但没有办法把整个场景描述出来。

2. Lack elegance of generative: Priors, 结构, 不确定性

3. Alternative notions of penalty functions, regularization, 核函数

4. 黑盒操作: 变量间的关系不清楚，不可视

**常见的机器学习方法**：

1.  logistic regression

2. 支持向量机（SVM）

3. 传统的神经网络（traditional neural networks）

4. K近邻，最近邻（Nearest neighbor）

5. Conditional random fields(CRF): 目前最新提出的热门模型，从NLP领域产生的，正在向ASR和CV上发展。

######主要应用：

1. 图像文本分类

2. 生物科学分析

3. 时间序列预测

### 生成模型Generative Model

估计的是联合概率分布（joint probability distribution），$$p(class, context)=p(class|context)*p(context)$$,换用之前的描述就是$$p(y, x)=p(y|x)*p(x)$$.

用于随机生成的观察值建模，特别是在给定某些隐藏参数情况下。在机器学习中，或用于直接对数据建模（用概率密度函数对观察到的draw建模），或作为生成条件概率密度函数的中间步骤。通过使用**贝叶斯定律**可以从生成模型中得到条件分布。

如果观察到的数据是完全由生成模型所生成的，那么就可以拟合生成模型的参数，从而仅可能的增加数据相似度。但观测数据往往完全从生成模型得到，所以比较准确的方式是直接对条件密度函数建模，即使用分类或回归分析。

######主要特点:

1. 一般主要是对**后验概率**建模，从统计的角度表示数据的分布情况，能够反映同类数据本身的相似度

2. 只关注自己的inclass本身（即点左下角区域内的概率），不关心到底判别边界在哪


######优点:

1. 模型可以通过增量学习得到（同意！）

2. 研究单类问题比判别模型灵活性强(怀疑？)

3. 能用于数据不完整（missing data）情况(怀疑？)

4. 实际上带的信息要比判别模型丰富（多太多！同意！）

5. prior knowledge can be easily taken into account（同意！）

6. modular construction of composed solutions to complex problems

7. robust to partial occlusion and viewpoint changes

8. can tolerate significant intra-class variation of object appearance

######缺点:

1. 学习和计算过程比较复杂

**常见的机器学习方法**：

1. **Gaussians判别分析**

2. ** Naive Bayes**， Bayesian networks

3. Mixtures of Gaussians（混合高斯模型）

4. HMMs，Markov random fields

5. Sigmoidal belief networks

####两者之间的关系

**由生成模型可以得到判别模型，但由判别模型得不到生成模型。**

>>  参考

http://prfans.com/forum/viewthread.php?tid=80

http://hi.baidu.com/cat_ng/blog/item/5e59c3cea730270593457e1d.html

http://en.wikipedia.org/wiki/Generative_model

http://blog.csdn.net/yangleecool/archive/2009/04/05/4051029.aspx