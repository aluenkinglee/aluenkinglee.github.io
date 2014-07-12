---
layout: post
title: "use svm to classify whether a person is good or bad on credit using sns data"
date: 2014-07-12 17:21
comments: true
categories: 机器学习 svm 金融信用 libsvm
---

首先说明这只是一个思路方向性的大概说明介绍，更多关于业务方面的内容不方便介绍。

如何评价一个人的金融信用，这个可以搜集用户的一些基本数据，比如职业，大学，社交数据来进行评分判断，至于为什么选取这些特征，一个很显然的理由就是
一个人越倾向使用社交应用，那么这个人就越可能是真实的，可信的。职业学校和人的信用也是成一定关系但不是绝对就是向我们想当然的那样。

之前的文章也断断续续的聊过伪匹配，分类的内容，是和此有一定关联。

对于模型的构建，方式有很多，贝叶斯网络，逻辑回归模型，svm模型，甚至是神经网络模型都可以对此进行建模使用，但是哪个性能更好呢？不知道，只有分别做出之后
比较才可以说明问题。

<!-- more -->

#### 非线性SVM

前面的blog有讲到线性svm，对于非线性分类器就要把x映射到特征空间,同时考虑误差ε的存在（即有些样本点会越过分类边界），上述优化问题变为：、

$$
\min { \quad \frac { 1 }{ 2 } \left\| w \right\| ^{ 2 }+c\sum _{ i=1 }^{ l }{ { \xi  }_{ i } }  } \\ st.\quad y_{ i }(\omega ^{ T }\phi (x_i)-b)\ge 1-\xi _{ i },\left( \xi _{ i }>0 \right) 
$$

从输入空间是映射到特征空间的函数称为核函数，LibSVM中使用的默认核函数是RBF（径向基函数radial basis function），即

$$
K(x,y)=exp{ \left( \frac { -\left\| x-y \right\| ^{ 2 } }{ 2\sigma ^{ 2 } }  \right)  }
$$

这样一来就有两个参数需要用户指定：c和gamma。实际上在LibSVM中用户需要给出一个c和gamma的区间，
LibSVM采用交叉验证cross-validation accuracy的方法确定分类效果最好的c和gamma。

举个例子说明什么是交叉验证，假如把训练样本集拆成三组，然后先拿 1 跟 2 来 train model 并 predict 3 以得到正确率；
再来拿 2 跟 3 train 并 predict 1 ，最后 1,3 train 并 predict 2 ，最后取预测精度最高的那组c和gamma。

####libsvm

点击[here](http://www.csie.ntu.edu.tw/~cjlin/cgi-bin/libsvm.cgi?+http://www.csie.ntu.edu.tw/~cjlin/libsvm+tar.gz)下载libsvm.

看readme即可使用了。


LibSVM要求处理的文件数据都满足如下格式：

```text
rlabel1 index1:value1   index2:value2   …...
rlabel2 index1:value1   index2:value2   …...
```

rlabel表示分类，为一个数字。Index从1开始递增，表示输入向量的序号，value是输入向量相应维度上的值，如果value为0,该项可以不写。下面是一个示例文件：

目前简单的分类所使用的属性有：年龄WOE   类型WOE   信用卡邮箱授权WOE  搜多引擎返回数WOE  贴吧搜索返回数WOE  微博数WOE  活跃度WOE  淘友朋友WOE 人人看过的人数WOE  好友数WOE  微博注册时间WOE
预测值为还款情况（还款1，逾期0）

```text
0 1:0.010003 2:-0.10524 3:0.131117521 4:0.061095504 5:0.029130009 6:-0.178635093 7:-0.149465309 8:-0.03275873 9:-0.047039163 10:-0.047039163 11:-0.158328769
0 1:0.010003 2:-0.10524 3:0.131117521 4:-0.025995297 5:-0.051360537 6:-0.090148625 7:-0.149465309 8:-0.005168969 9:0.000132475 10:-0.050262447 11:-0.158328769
0 1:-0.08606 2:-0.10524 3:0.131117521 4:-0.163782716 5:-0.051360537 6:-0.178635093 7:-0.149465309 8:-0.03275873 9:-0.047039163 10:-0.047039163 11:-0.158328769
0 1:-0.21088 2:0.009167 3:0.131117521 4:0.061095504 5:0.029130009 6:-0.090148625 7:-0.129577755 8:-0.03275873 9:0.000132475 10:-0.050262447 11:-0.226676962
```

svm_scale用于把输入向量按列进行规范化（或曰缩放）。

```bash
Usage: svm-scale [options] data_filename
options:
-l lower : x scaling lower limit (default -1)
-u upper : x scaling upper limit (default +1)
-y y_lower y_upper : y scaling limits (default: no y scaling)
-s save_filename : save scaling parameters to save_filename
-r restore_filename : restore scaling parameters from restore_filename
```

例如： `svm_scale -l 0 -u 1 -s range trainSet > trainSet.scale`则输入文件是trainSet，输出文件是trainSet.scale，把输入向量的各列都缩放到[0，1]的范围内，range文件中保存了相关的缩放信息。

这个时候我们应该把训练集分为两部分，训练集和测试集，在训练集上通过交叉验证学到最佳的参数，然后在测试集上验证。

```bash
$python subset.py
Usage: subset.py [options] dataset number [output1] [output2]
This script selects a subset of the given data set.
options:
-s method : method of selection (default 0)

     0 -- stratified selection (classification only)

     1 -- random selection

output1 : the subset (optional)

output2 : the rest of data (optional)

If output1 is omitted, the subset will be printed on the screen.

$python subset.py trainSet 0.3*m(实例个数) trainSet.data testSet.data
$ ./tools/subset.py ./trainningset.txt 280 testSet trainSet
```

grid.py是一种用于RBF核函数的C-SVM分类的参数选择程序。用户只需给定参数的一个范围，grid.py采用交叉验证的方法计算每种参数组合的准确度来找到最好的参数。

```python
python grid.py
Usage: grid.py [-log2c begin,end,step] [-log2g begin,end,step] [-v fold]
       [-svmtrain pathname] [-gnuplot pathname] [-out pathname] [-png pathname]
       [additional parameters for svm-train] dataset
The program conducts v-fold cross validation using parameter C (and gamma)= 2^begin, 2^(begin+step), ..., 2^end.
```

首先`sudo apt-get install gnuplot`

然后编译C++版本的LibSVM，生成svm-train二进制可执行文件。

```bash
python grid.py -log2c -5,5,1 -log2g -4,0,1 -v 5 -svmtrain /path/to/your/svm-train -m 500 trainSet.data (svm-train 的路径自个找好)
```

-m 500是使用svm_train时可以使用的参数。
最后输出两个文件：dataset.png绘出了交叉验证精度的轮廓图，dataset.out对于每一组log2(c)和log2(gamma)对应的CV精度值。

得到c=16，g=1

这个是我实验的截图

![实验截图](https://raw.githubusercontent.com/aluenkinglee/mlclass/master/libsvm-3.18/trainSet.png)


最后来训练我们的模型

```bash
$ ./svm_train

-s svm_type : set type of SVM (default 0)
0 -- C-SVC
-t kernel_type : set type of kernel function (default 2)
0 -- linear: u'*v
2 -- radial basis function: exp(-gamma*|u-v|^2)
-g gamma : set gamma in kernel function (default 1/num_features) num_features是输入向量的个数
-c cost : set the parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1)
-m cachesize : set cache memory size in MB (default 100) 使用多少内存
-e epsilon : set tolerance of termination criterion (default 0.001) 
-h shrinking : whether to use the shrinking heuristics, 0 or 1 (default 1) 
-wi weight : set the parameter C of class i to weight*C, for C-SVC (default 1) 当各类数量不均衡时为每个类分别指定C
-v n: n-fold cross validation mode交叉验证时分为多少组
-q : quiet mode (no outputs)

$ svm_train -s 0 -c 16 -t 2 -g 1 -e 0.01 trainSet.scale
$ ./svm-train -s 2 -c 16 -g 1 -v 5 ./trainSet 

```

会得到训练结果，然后使用这个模型来预测测试集的数据准确性


```bash
./svm-predict testSet result
./svm-predict -b 0 testSet trainSet.model  result
```

### 实验结果

对数据进行格式转换后，首先对数据集进行分割得到训练集和测试集，选择常规比例为6：4

1.一开始未设置最有参数时即C和simga的值时，由训练集得到的模型分类性能在测试集上达到了Accuracy = 79.64285714285714% (223/280)的精度。

2.选择合适的参数。

通过交叉验证方法对模型进行选择得到针对训练集上最优的参数为c=16 g=1

由此得到的训练模型在测试集上的准确率达到了Accuracy = 81.4286% (228/280) (classification)，将近2%的精度提升，也算不错了。

目前由这些属性判断用户的还款情况，准确率在80%左右。但是预测结果全部预测为1.不合理。

实验结果准确率，在指定one-vs-class 之后，准确率为40%。比起逻辑回归模型仍然好点。

>> Ref

数据可以到[这里](https://github.com/aluenkinglee/mlclass/tree/master/libsvm-3.18)
