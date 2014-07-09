---
layout: post
title: "拉格朗日对偶"
date: 2014-06-03 21:41
comments: true
categories: 机器学习 拉格朗日对偶
---

在解决约束最优化的额问题中，常常利用拉格朗日对偶性将原始问题转化为对偶问题，通过解对偶问题得到原始问题的解。该方法主要应用的SVM和最大熵模型中。

先抛开上面的二次规划问题，先来看看存在等式约束的极值问题求法，比如下面的最优化问题：

$$
\min _{ w }{ f\left( w \right)  } \\ s.t\quad h_{ i }\left( w \right) =0,\quad i=1,\cdots ,l
$$
         
目标函数是f(w)，下面是等式约束。通常解法是引入拉格朗日算子，这里使用$$\beta$$来表示算子，得到拉格朗日公式为

<!-- more -->

$$
L \left( w,\beta  \right) =f\left( w \right) +\sum _{ i=1 }^{ l }{ \beta _{ i }h_{ i } } \left( w \right) 
$$

 L是等式约束的个数。

然后分别对w和 $$\beta$$求偏导，使得偏导数等于0，然后解出w和 $$\beta_i$$。至于为什么引入拉格朗日算子可以求出极值，原因是f(w)
的dw变化方向受其他不等式的约束，dw的变化方向与f(w)的梯度垂直时才能获得极值，而且在极值处，f(w)的梯度与其他等式梯度的线性组合平行，因此他们之间存在线性关系。（参考《最优化与KKT条件》）
然后我们探讨有不等式约束的极值问题求法，问题如下：

$$
\min _{ w }{ f\left( w \right)  } \\ 
s.t\quad h_{ i }\left( w \right) =0,\quad i=1,\cdots ,l\\ 
\quad \quad \quad g_{ i }\left( w \right) \le 0,\quad i=1,\cdots ,k
$$

我们定义一般化的拉格朗日公式

$$
L\left( w,\beta  \right) =f\left( w \right) +\sum _{ i=1 }^{ k }{ \alpha _{ i }g_{ i } } \left( w \right) \sum _{ i=1 }^{ l }{ \beta _{ i }h_{ i } } \left( w \right) 
$$
 
这里的$$\alpha _{ i } $$和$$\beta _{ i } $$ 都是拉格朗日算子。如果按这个公式求解，会出现问题，因为我们求解的是最小值，而这里的 $$g_{ i } \left( w \right) $$已经不是0了，我们可以将 $$\alpha _{ i } $$调整成很大的正值，来使最后的函数结果是负无穷。因此我们需要排除这种情况，我们定义下面的函数：

$$
{ \theta  }_{p  }\left( w \right) =\max _{ \alpha ,\beta :{ \alpha  }_{ i }\ge 0 }{ L\left( w,\alpha ,\beta  \right)  } 
$$

这里的P代表primal。假设$$g_{ i }\left( w \right)$$ >0 或者$$h_{ i }\left( w \right) \neq 0$$ ，那么我们总是可以调整 $$\alpha_i$$和$$\beta_i$$ 来使得$$ { \theta  }_{ p  }\left( w \right)$$有最大值为正无穷。而只有g和h满足约束时，$$ { \theta  }_{ p }\left( w \right)$$ 为f(w)。这个函数的精妙之处在于 $$\alpha_i \ge 0$$，而且求极大值。

因此,

$$
{ \theta  }_{ p  }\left( w \right) =\begin{cases} f\left( w \right) ,\quad w满足原始问题约束 \\ +\infty ，其他\quad  \end{cases}
$$

所以如果考虑极小化问题

$$
{ \min _{ w }{ { \theta  }_{p  }\left( w \right)  }  }={ \min _{ w }{ \max _{ \alpha ,\beta :{ \alpha  }_{ i }\ge 0 }{ L\left( w,\alpha ,\beta  \right)  }  }  }
$$

它是原始问题的等价解，原始最优化问题转化成了拉格朗日函数的极小极大问题，这时候把原始问题的最优值记为：

$$
{ p }^{ \ast  }=\min _{ w }{ { \theta  }_{ p }\left( w \right)  } 
$$

哎哟，看看我们的等价形式哦，首先有两个参数，其中还是一个不等式约束=。=,考虑下对偶吧，极小极大问题转化为等价的极大极小问题。

####对偶形式

定义$${ \theta  }_{ D }\left( \alpha ,\beta  \right) =\min _{ w }{ L\left( w,\alpha ,\beta  \right)  } $$，在考虑极大化$${ \theta  }_{ D }\left( \alpha ,\beta  \right) $$,先把这两个参数看成固定值，求关于w的最小值，之后在求对偶的最大值即

$$
\max _{ \alpha ,\beta :{ \alpha  }_{ i }\ge 0 }{ { \theta  }_{ D }\left( \alpha ,\beta  \right)  } =\max _{ \alpha ,\beta :{ \alpha  }_{ i }\ge 0 }{ \min _{ w }{ L\left( w,\alpha ,\beta  \right)  }  } 
$$

 这个问题是原问题的对偶问题，相对于原问题只是更换了min和max的顺序，而一般更换顺序的结果是Max Min(X) <= MinMax(X)。然而在这里两者相等。用$${ d }^{ \ast  }$$ 来表示对偶问题如下：

$$
{ d }^{ \ast  }=\max _{ \alpha ,\beta :{ \alpha  }_{ i }\ge 0 }{ \min _{ w }{ L\left( w,\alpha ,\beta  \right)  }  } \le { \min _{ w }{ \max _{ \alpha ,\beta :{ \alpha  }_{ i }\ge 0 }{ L\left( w,\alpha ,\beta  \right)  }  }  }={ p }^{ \ast  }
 $$

 存在 $${ w }^{ \ast  },{ \alpha }^{ \ast  },{ \beta }^{ \ast  }$$ 使得$${ w }^{ \ast  }$$是原问题的解，$${ \alpha }^{ \ast  },{ \beta }^{ \ast  }$$  是对偶问题的解。还有 $${ p }^{ \ast  }={ d}^{ \ast  } = L({ w }^{ \ast  },{ \alpha }^{ \ast  },{ \beta }^{ \ast  })$$ 

 另外，  $${ w }^{ \ast  },{ \alpha }^{ \ast  },{ \beta }^{ \ast  }$$ 满足库恩-塔克条件（Karush-Kuhn-Tucker, KKT condition），该条件如下：

![库恩-塔克条件（Karush-Kuhn-Tucker, KKT condition）](https://github.com/aluenkinglee/aluenkinglee.github.io/blob/source/source/images/2014-06-03-lagrange-duality/kkt.png?raw=true 库恩-塔克条件（Karush-Kuhn-Tucker, KKT condition）")

 所以如果 $${ w }^{ \ast  },{ \alpha }^{ \ast  },{ \beta }^{ \ast  }$$  满足了库恩-塔克条件，那么他们就是原问题和对偶问题的解。当$$g_{i}(w^{ \ast  })=0$$ 时，w处于可行域的边界上，这时才是起作用的约束。而其他位于可行域内部（ $$g_{i}(w^{ \ast  })<0$$ 的）点都是不起作用的约束，其 $${ \alpha }^{ \ast  }=0$$。这个KKT双重补足条件会用来解释支持向量和SMO的收敛测试。

 KKT的总体思想是将极值会在可行域边界上取得，也就是不等式为0或等式约束里取得，而最优下降方向一般是这些等式的线性组合，其中每个元素要么是不等式为0的约束，要么是等式约束。对于在可行域边界内的点，对最优解不起作用，因此前面的系数为0。


>> 参考

1. Andrew Ng的原始课件讲义

2. 统计学习方法