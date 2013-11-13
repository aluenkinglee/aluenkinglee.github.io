---
layout: post
title: "virtualbox网络环境和登录"
date: 2013-10-20 22:46
comments: true
categories: SDN virtualbox SSH
---

最近由于实验室需要研究SDN的一些东西，再此先来搭配下环境。
对于虚拟机的一些选择，可以看这里:
[vm-setup-notes](http://mininet.org/vm-setup-notes/) 
我这里的环境是:

`os:            debian 7(wheezy) 64bit`

`xwindows:      kde`

`VM：           virtualbox`

`controller:    pox`

按照上述的网址，可以比较容易的搭配出来需要的环境，下面就说以下之后遇到的问题。

* virtualbox的鼠标独占问题，只是提示太不明明显了！对于那个right ctrl的提示我郁闷了好一会，我以为是鼠标右键+ctrl...其实是右边的ctrl键。对于这个键的设定，你可以从“管理”>“全局设定”>"热键"来设置你喜欢的按键..我就吐槽这么一句，之前vmware对此的说明很明了！
* Using Host-only adapter to SSH guest from host.

### NAT模式
特点：
1. 如果主机可以上网，虚拟机可以上网
2. 虚拟机之间不能ping通
3. 虚拟机可以ping通主机（此时ping虚拟机的网关，即是ping主机）
4. 主机不能ping通虚拟机

但是这个情况下是不能满足我的使用条件...我需要在host登录到guest中，so pass

### 桥接模式
特点：
1. 如果主机可以上网，虚拟机可以上网
2. 虚拟机之间可以ping通
3. 虚拟机可以ping通主机
4. 主机可以ping通虚拟机 (以上各点基于一个前提：主机可以上网)
5. 如果主机不可以上网，所有1-4特点均无

<!--more-->

这个挺好，觉得可以满足我的要求但是，发现它的IP地址要求是自动非配的，我们实验室的IP是校园内网固定IP...

### Host-only Adapter模式

特点：

1. 虚拟机`不`可以上网
2. 虚拟机之间可以ping通
3. 虚拟机可以ping通主机（注意虚拟机与主机通信是通过主机的名为VirtualBox Host-Only Network的网卡，因此ip
是该网卡ip `192.168.56.1`，而不是你现在正在上网所用的ip）
4. 主机可以ping通虚拟机

应用场景：
需要搭建一个模拟局域网，所有机器可以互访。（颇不得以选这个的..）

ip样式：`netstat -rn`查看路由表
ip 与本机VirtualBox Host-Only Network的网卡ip在同一网段内（默认192.168.56.*）
网关 本机VirtualBox Host-Only Network的网卡ip（默认192.168.56.1）


### 登录虚拟机
登录虚拟机，首先开开virtualbox的虚拟机

```
mininet-vm login: mininet
Password: mininet
```

接下来查看我们的vm地址：（在guest里面操作）

```
sudo ifconfig
```
例如我这里的ip地址是`192.168.56.101`,然后就使用我们可爱的SSH

```
ssh -Y mininet@192.168.56.101
```
输入密码就可以了

当然上面的不好看，做下别名好了。（在我们host里面操作）

```
sudo nano /etc/hosts
192.168.56.101 mininet-vm
```

保存即可，这样子就可以使用

```
ssh -Y mininet@mininet-vm
```

接下来就可以去玩下面的实验了^^

> 参考文档:

> [VirtualBox虚拟机网络环境解析](http://blog.csdn.net/yxc135/article/details/8458939)

> [ssh命令用于远程登录上Linux主机](http://www.live-in.org/archives/832.html)

> [SSH into VM](http://mininet.org/vm-setup-notes/)