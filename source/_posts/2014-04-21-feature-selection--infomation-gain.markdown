---
layout: post
title: "特征选择-信息增益算法"
date: 2014-04-21 15:41
comments: true
categories: 特征选择 信息增益算法
---

**信息增益**：特征 $A$ 对于训练数据集$D$的信息增益 $g\left( D,A \right)$,定义为集合 $D$ 的经验熵 $H\left( D \right)$ 与特征 $A$ 在给定条件下 $D$ 
的经验条件熵c之差。
<!--more-->
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
