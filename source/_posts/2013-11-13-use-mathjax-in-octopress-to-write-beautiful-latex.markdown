---
layout: post
title: "Use MathJax in Octopress to Write beautiful LaTex"
date: 2013-11-13 12:37
comments: true
categories: LaTex Octopress
---
#### 首先看下$\LaTeX$的例子

* 块状的$\LaTeX$ 数学公式 (1)

$$
\begin{align*}
  & \phi(x,y) = \phi \left(\sum_{i=1}^n x_ie_i, \sum_{j=1}^n y_je_j \right)
  = \sum_{i=1}^n \sum_{j=1}^n x_i y_j \phi(e_i, e_j) = \\
  & (x_1, \ldots, x_n) \left( \begin{array}{ccc}
      \phi(e_1, e_1) & \cdots & \phi(e_1, e_n) \\
      \vdots & \ddots & \vdots \\
      \phi(e_n, e_1) & \cdots & \phi(e_n, e_n)
    \end{array} \right)
  \left( \begin{array}{c}
      y_1 \\
      \vdots \\
      y_n
    \end{array} \right)
\end{align*}
$$

<!--more-->

* 块状的$\LaTeX$ 数学公式 (2)

另外一个例子[Multivariate normal distribution](http://en.wikipedia.org/wiki/Multivariate_normal_distribution "Multivariate normal distribution"):

$$
f_x(x_1,...,x_k)=\frac{1}{(2\pi)^{k/2}|\Sigma|^{1/2}}exp(-\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu))
$$

* 内联$\LaTeX$ 数学公式 (3)

在段内插入LaTeX代码： $\exp(-\frac{x^2}{2})$ 。

#### $\LaTeX$ support 

为了能够使用MathJax对数学公式渲染，还是得使用几个步骤的。

* 添加ramdown组件

```
gem install kramdown
```

* 找到`source/_includes/custom/head.html`这个文件，加入以下内容

```
<script type="text/x-mathjax-config">
MathJax.Hub.Register.StartupHook("TeX Jax Ready",function () {
          MathJax.InputJax.TeX.prefilterHooks.Add(function (data) {
                  data.math = data.math.replace(/^\s*<!\[CDATA\[\s*((?:\n|.)*)\s*\]\]>\s*$/m,"$1");
                    });
          });
</script>

<script type="text/x-mathjax-config">
  MathJax.Hub.Config({
    tex2jax: {
      inlineMath: [ ['$','$'], ["\\(","\\)"] ],
      processEscapes: true
    }
  });
</script>

<script type="text/x-mathjax-config">
    MathJax.Hub.Config({
      tex2jax: {
        skipTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code']
      }
    });
</script>

<script type="text/x-mathjax-config">
    MathJax.Hub.Queue(function() {
        var all = MathJax.Hub.getAllJax(), i;
        for(i=0; i < all.length; i += 1) {
            all[i].SourceElement().parentNode.className += ' has-jax';
        }
    });
</script>

<script type="text/javascript"
   src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
</script>

```

* 修改_config.yml 文件 把markdown的渲染引擎从`rdiscount` 变为`kramdown`

* 对于内联的$\LaTeX$ 数学公式，只需简单使用`$`...`$`即可。

就像这样：

```
在段内插入LaTeX代码： $\exp(-\frac{x^2}{2})$ 。
```


* 对于块状的$\LaTeX$ 数学公式，只需简单使用`$$`...`$$`即可。
就像这样：

```
$$
f_x(x_1,...,x_k)=\frac{1}{(2\pi)^{k/2}|\Sigma|^{1/2}}exp(-\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu))
$$
```

#### Reference

[在octopress中使用latex](http://kkx.github.io/blog/2012/05/05/zai-octopresszhong-shi-yong-latex/ "在octopress中使用latex")

[Hello world](http://kqueue.org/blog/2012/01/05/hello-world/ "Hello world")

[在Octopress中使用Latex](http://hungmingwu-blog.logdown.com/posts/14279-latex-on-octopress "在Octopress中使用Latex")

[在Octopress中使用Latex](http://yanping.me/cn/blog/2012/03/10/octopress-with-latex/ "在Octopress中使用Latex")

[Markdown 语法说明 (简体中文版)](http://wowubuntu.com/markdown/#p "Markdown 语法说明 (简体中文版)")

[kramdown Syntax](http://kramdown.gettalong.org/syntax.html#math-blocks "kramdown Syntax")
