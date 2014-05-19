---
layout: post
title: "generative method"
date: 2014-04-30 22:52
comments: true
categories: 机器学习 Machine learning 整理 生成学习
---

线性回归模型和logistic回归是判别模型，也就是根据特征值来求结果的概率。形式化表
示为$$p(y|x;\theta)$$在参数$$\theta$$确定的情况下，求解条件概率$$p(y|x)$$。
通俗的解释为在给定特征后预测结果出现的概率。

就按照Andrew Ng讲的那样，确定肿瘤是良性的还是恶性的，可以使用判别模型的方法是先
从历史数据中学习到模型，然后通过提取肿瘤的特征来预测出它是良性恶性的概率。

反过来，要是我们先从良性肿瘤学习出良性肿瘤的模型，从恶性肿瘤学习出恶性肿瘤的模型，
然后提取肿瘤的特征，放到良性肿瘤的模型看下概率，在放到恶性肿瘤的模型看下概率，哪个
大是哪个。

形式化表示为求$$P(X|Y)$$,x是特征，y是类型即模型。
利用贝叶斯公式发现两个模型的统一性：

$$
p(y|x)=\frac { p(x|y)p(y) }{ p(x) } 
$$

由于我们关注的是 y 的离散值结果中哪个概率大（比如良性肿瘤和恶性肿瘤哪个概率大），
而并不是关心具体的概率，因此上式改写为：

$$
\quad \quad \quad \quad \begin{eqnarray} \max _{ y }{ p(y|x) }  & = & \max _{ y }{ \frac { p(x|y)p(y) }{ p(x) }  }  \\  & = &\max _{ y } p(x|y)p(y) \end{eqnarray}
$$


其中$$p(x|y)$$称为后验概率,$$p(y)$$称为先验概率.

由$$p(x|y)*p(y)=p(x,y)$$,因此有时称判别模型求的是条件概率，生成模型求的是联
合概率。

常见的判别模型有线性回归、对数回归、线性判别分析、支持向量机、boosting、条件
随机场、神经网络等。

常见的生产模型有隐马尔科夫模型、朴素贝叶斯模型、高斯混合模型、LDA、Restricted 
Boltzmann Machine 等。

上篇博客较为详细地介绍了两个模型

###高斯判别分析（Gaussian discriminant analysis）

##### 多维正太分布

多变量正态分布描述的是n维随机变量的分布情况。所以这里的$$\mu $$变成了n维随机变量，$$\sigma $$也变成了
矩阵$$\Sigma $$.记做$$N(\mu,\Sigma)$$.假设有 n 个随机变量$$X_1,X_2,\cdots ,X_n$$.所以显而易见，$$\mu $$的第i个分量是$$E(X_i),\Sigma_{ii}=Var(X_i),\Sigma_{ij}=Cov(X_i,X_j)$$.

概率密度函数如下：

$$
p(x;\mu,\Sigma)=\frac { 1 }{ \left( 2\pi  \right) ^{ n/2 }\left| \Sigma  \right| ^{ 1/2 } } exp\left( -\frac { 1 }{ 2 } \left( x-\mu \right)^T \Sigma^{-1}{\left(x-\mu\right)} \right) 
$$

##### 模型分析与应用

如果输入特征x连续型随机变量，那么可以使用高斯判别分析模型来确定$$p(x|y)$$。模型如下,先以二元分布即伯努利分布来说（因为前面的例子是二元的）:

$$
y\sim Bernoulli\left( \phi \right) \\
x|y=0\sim N(\mu_0,\Sigma)\\
x|y=1\sim N(\mu_1,\Sigma)
$$


