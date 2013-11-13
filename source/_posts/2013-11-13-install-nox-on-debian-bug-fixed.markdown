---
layout: post
title: "install nox on debian-bug fixed"
date: 2013-11-13 22:44
comments: true
categories: SDN nox
---

首先需要声明的一点是，到目前（2013/11）Nox在所有SDN控制器中不是最火的（相比Pox，floodlight），但是作为最先开发的一个SDN而言，还是有研究意义的。

<!--more-->

##### 安装提示

```
cd /etc/apt/sources.list.d/
sudo wget http://openflowswitch.org/downloads/debian/nox.list
sudo apt-get update
sudo apt-get install nox-dependencies
sudo apt-get install libtbb-dev
sudo apt-get install libboost-serialization-dev libboost-all-dev
```

然后在你想放置Nox源码的地方做如下操作：

```
git clone git://github.com/noxrepo/nox
cd nox
./boot.sh
mkdir build
cd build
../configure
make -j 5
make install
```

很不幸，遇到了如下的错误：

```
../../src/builtin/component.cc:414:1: required from here
/usr/include/boost/property_tree/detail/json_parser_read.hpp:105:17: error:
no matching function for call to ‘boost::property_tree::basic_ptree
std::basic_string<char, std::basic_string >::push_back(std::pair
std::basic_string<char, std::basic_string >)’
/usr/include/boost/property_tree/detail/json_parser_read.hpp:105:17: note:
candidate is:
In file included from /usr/include/boost/property_tree/ptree.hpp:516:0,
from ../../src/include/component.hh:35,
from ../../src/builtin/component.cc:18:
/usr/include/boost/property_tree/detail/ptree_implementation.hpp:362:9:
note: boost::property_tree::basic_ptree::iterator
boost::property_tree::basic_ptree::push_back(const value_type&) [with Key =
std::basic_string; Data = std::basic_string; KeyCompare = std::less
std::basic_string<char >; boost::property_tree::basic_ptree::value_type =
std::pair, boost::property_tree::basic_ptreestd::basic_string<char,
std::basic_string > >]
/usr/include/boost/property_tree/detail/ptree_implementation.hpp:362:9:
note: no known conversion for argument 1 from ‘std::pair
std::basic_string<char, std::basic_string >’ to ‘const value_type& {aka
const std::pair, boost::property_tree::basic_ptreestd::basic_string<char,
std::basic_string > >&}’
make[4]: *** [nox_core-component.o] Error 1
```

这个错误来源于nox依赖的boost库版本（1.49）的错误.注意错误原因是：

```
/usr/include/boost/property_tree/detail/json_parser_read.hpp:105
error:
no matching function for call to ‘boost::property_tree::basic_ptree
std::basic_string<char, std::basic_string >::push_back(std::pair
std::basic_string<char, std::basic_string >)’
```

同样的问题点击[这里](http://lists.noxrepo.org/pipermail/nox-dev-noxrepo.org/2013-February/000668.html '[nox-dev] Nox build fails')

解决方法：`/usr/include/boost/property_tree/detail/json_parser_read.hpp`找到这个文件。定位到`105行，`

```
c.stack.back()->push_back(std::make_pair(c.name, Str(b, e)));
```

换成

```
c.stack.back()->push_back(std::make_pair(c.name, Ptree(Str(b, e))));
```

重新make&& make install 即可。

##### Reference

1. [read_json does not compile on GCC 4.7.0 with std=c++11](https://svn.boost.org/trac/boost/ticket/6785)

2. [nox-dev Nox build fails](http://lists.noxrepo.org/pipermail/nox-dev-noxrepo.org/2013-February/000668.html)
