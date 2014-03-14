---
layout: post
title: "Python中得一些重要的细节"
date: 2013-03-15 16:17
comments: true
categories: python
---

字符串可以被下表索引，和C一样，第一个字符同样是0。

Python中并没有单独的字符类型，一个字符就是长度为1的字符串。

和C字符串不同的是，python字符串值不能更改。

<!-- more -->

```python
>>> word = 'string'
>>> word[3]='d'
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: 'str' object does not support item assignment
```

关于unicode的一些说明，encode(),unicode()那些往
[这里看](http://docs.python.org/2/tutorial/introduction.html#unicode-strings)

python中一些iterable types（具体术语不知道怎么称呼）是分成可以`immutable`和`mutable`的：

#####可更改的（`mutable`）:

* 字典型(dictionary)

* 列表型(list)

#####不可更改（mutable）：

* 元组（tuple)

* 数值型（number）

* 字符串(string)

所有的切片操作都是返回一个新的包含请求元素的list。

```python
>>> a = ['spam', 'eggs', 100, 1234]
>>> a[:]
['spam', 'eggs', 100, 1234]
```

pass语句请看[这里](http://docs.python.org/2/reference/simple_stmts.html#pass)

函数中的变量，函数中得变量赋值操作都是把值存在`本地符号表（local symbol table）`中，
变量引用时先查看本地符号表，在然后是`全局符号表（global symbol table）`，
最后才是`内置表（table of built-in names）`，全局变量得先申明下，在使用

```python
v= 1
def test_global_vars():
    """ 测试使用global"""
    #global v += 1 #wrong!!
    global v 
    v += 1
    print v
```

突然想到了python中得变量类型这些东西，以下是实验

```python
def test_global_vars():
    """ 测试使用global"""
    global v
    v += 1
    print v,hex(id(v))
 
def foo(n):
    """测试实参是怎么回事，by value，value总是个函数引用，不是那个
    对象的值,实参也是跟函数内的变量一样被放入到了本地符号表"""
    n += 1
    print n,hex(id(n))
 
v= 1
print v,hex(id(v))
test_global_vars()
print v,hex(id(v))
foo(v)
print v,hex(id(v))
v+=1
print v,hex(id(v))
```

这是我机器上的一次run result

```
1 0x1429168
2 0x142915c
2 0x142915c
3 0x1429150
2 0x142915c
3 0x1429150
```

会发现v的值一样的时候，id值都是一样的。


```python
v = 1 			#将名字 v 与内存中值为1的内存绑定在一起。
test_global_vars() 	#这句话之后就是把v 和 内存中为2的地址绑定在一起。所以地址会变。
foo(v) 			#形参n 和 内存中为 3 的地址绑定在一起。实参v还是绑定在 2 的地址。
v += 1 			#v的值改变为3 所以地址就绑定到了 3 在内存中得地方。所以这个时候会发现和上一句的地址一样。
```

很奇特，但是这个样子会有什么好处?

* 首先python支持函数编程，函数式编程在运算的过程中值肯定是不会变得。
想想我们数学中得变量（怎么可能会出现x = x + 1 ！！）数学上讲不通。
这对函数式编程是个利好消息。（但是这很明显不是函数式编程，应该是这
个特性在lamda演算中很有用，待解决）

* 这个…受C、C++的“毒害”深啊，目前还是转不过来,但是在看列表`可变`之后
，刹那间觉得列表这玩意像指针那种东西啊

```python
dic = {"value":3}
b = dic
print dic,b,hex(id(dic)),hex(id(b))
b["value"] = 5
print dic,b,hex(id(dic)),hex(id(b))
```

但是你看结果，他们都是绑定到同一块内存的，这种字典的就直接在源地址改了

```python
{'value': 3} {'value': 3} 0x1823d20 0x1823d20
{'value': 5} {'value': 5} 0x1823d20 0x1823d20
```
python就是这么设计的，至于为什么，不知道(待解决)


在说句题外话，你看django那些代码写的，都是使用的tuple之类的。url(省略号)…为什么会这个样子，来看看实验的结果

```python
class A(object):
    """docstring for A"""
    x = []
    # self.x = [] #这个是错误的语法，只能在__init__中定义
    def __init__(self, arg):
        #x和self.value有什么区别啊
        #x和self.x有什么区别啊
        super(A, self).__init__()
        self.arg = arg
        self.value = []
        self.value.append(1)
        print "in __init__"
        print "x address :\t\t\t\t", hex(id(x))
        print "self.x address :\t\t", hex(id(self.x))

    def set_x(self):
        self.x.append(1)
    def get_x(self):
        print "in get_x"
        print "x address :\t\t\t\t", hex(id(x))
        print "self.x address :\t\t", hex(id(self.x))
        return self.x
    def get_xx(self):#很诡异的存在啊，但这个只是定义的不同啊
        print "in get_xx"
        print "x address :\t\t\t\t", hex(id(x))
        print "self.x address :\t\t", hex(id(self.x))
        return x   

    def get_value(self):
        return self.value

for i in range(3):
    a = A("oj")
    a.set_x()
    #print A.__dict__
    print a.get_value()
    print a.get_xx()
    print a.get_x()
```

自己跑下试试，在__init__()内外定义的变量是不一样的，x，self.value类似static，全局共享。而self.x就不是了，在编码的时候一定要注意这种细节。

可能上面那个太复杂了些，那么看看这个。


```python
class b:
    x = []
    def __init__(self):
        self.y=[]
    def set(self):
        self.y.append(1)
        self.x.append(1)
    def get(self):
        return x,self.y,self.x

for i in range(3):
    a = b()
    a.set()
    print a.get(),b.__dict__
```

接下来还是关于函数的:

The default values are evaluated at the point of function definition in the defining scope, so that


```python
i = 5
def f(arg=i): # 默认的初始值 只 赋值一次 ，其后初始就不会改变了。
    print arg
i = 6
f()           # here will print  5！！ 所以这会为5 只要没传递新的参数
```

但是当形参是个可变对象 如 :列表，字典或者一些类的实例，这个时候会共享这个可变的对象！！！

虽然元组（Tuples）和列表（list）看起来很相似，但是使用的场景不同的。元组通常是不可变的，
通常包括类型不同的元素，可以通过解包和索引来访问。列表是可变的，通常都是些同一类型的元素迭代访问。

遍历技巧，可能需要下标的时候会用到 [这里](http://docs.python.org/2/tutorial/datastructures.html#looping-techniques)


---

_xxx      不能用’from module import *’导入

__xxx__ 系统定义名字

__xxx    类中的私有变量名

以单下划线开头_foo 的代表不能直接访问的类属性，需通过类提供的接口进行访问，不能用“from xxx import *”而导入；

以双下划线开头的__foo代表类的私有成员；以双下划线开头和结尾的__foo__代表python里特殊方法专用的标识，如__init__()代表类的构造函数。

“单下划线” 开始的成员变量叫做保护变量，意思是只有类对象和子类对象自己能访问到这些变量；

“双下划线” 开始的是私有成员，意思是只有类对象自己能访问，连子类对象也不能访问到这个数据。

######Python中的继承：

super可以避免显示的引用Base，这听起来很不错，最起码不用自己去判断使用哪个类的构造方法。
但是最最要的用途还是多继承问题。在python2.7中，得这样解决：super(thisClass,self).__init__() ,python3中变成了这样：super().__init__()

样例：仔细观察，不要忘记那个object！！

```python
class Point(object):
	def __init__(self,x=0,y=0):
		self.x = x;
		self.y = y;

class Circle(Point):
	"""docstring for Circle"""
	def __init__(self, radius=0,x=0,y=0):
		super(Circle,self).__init__()
		# Point.__init__(self,x,y)
		self.radius = radius;
c= Circle(1,0,0)

class Base(object):
    def __init__(self):
        print "Base created"

class ChildA(Base):
    def __init__(self):
        Base.__init__(self)

class ChildB(Base):
    def __init__(self):
        super(ChildB, self).__init__()

print ChildA(),ChildB()
```
























