---
layout: post
title: "git的安装和使用"
date: 2011-03-14 15:11
comments: true
categories: git  
---

git之前一直使用的，但是n久没用居然忘记怎么使用了。。。中途使用过程中居然还出现了

```bash
Permission denied (publickey).
fatal: The remote end hung up unexpectedly
```
<!--more-->

的错误，真是忘记怎么使用了，连SSH keys都没添加就想push啊…好吧，
真的是好记性不如烂笔头，老老实实的写下来已免在出这样的糗事


```bash
sudo apt-get install git git-core
ssh-keygen
```

然后系统提示输入文件保存位置等信息，连续敲三次回车即可，生成的SSH key文件保存在中～/.ssh/id_rsa.publickey

接着拷贝.ssh/id_rsa.pub文件内的所以内容，将它粘帖到github帐号管理中的添加SSH key界面中。

######打开github帐号管理中的添加SSH key界面的步骤如下

* 登录github

* 点击右上方的Accounting settings图标

* 选择 SSH key

* 点击 Add SSH key

在出现的界面中填写SSH key的名称，填一个你自己喜欢的名称即可，然后将上面拷贝的~/.ssh/id_rsa.pub文件内容粘帖到key一栏，
在点击“add key”按钮就可以了。添加过程github会提示你输入一次你的github密码。

###git 配置

git有三个配置文件，分别是`repo/.git/config`,`$HOME/.gitconfig`,`/etc/gitconfig`.

1. repo/.git/config 库级别的配置文件，只对当前库有效，优先级最高(git config –local)

2. $HOME/.gitconfig 用户级别的配置文件，对当前用户有效，优先级次之(git config –global)

3. /etc/gitconfig 系统全局配置文件，对整个系统有效，优先级最低(git config –system)

git config –list 可以查看当前的git配置列表

如果已经配置了，则会看到user.name 和 user.email的配置信息

如果没有,一般情况下在git提交时会使用机器名，诸如：unknown dev@xxx-PC.(none) 等类型的Author信息，肯定不方便了。

建议都配置明确的user.name 和 user.email信息。

#####可以通过下面的命令进行配置

```bash
git config user.name xxx
git config user.email xxx@xxx.com
```

配置完成后可以通过 git config –list 查看到.

这个准备工作算是完成了，其他的，就参考别人写得。在这不再写了。

>> 参考

1. [说明很详细](http://blog.sina.com.cn/s/blog_55465b470100s63h.html )

2. [这个有可能会用的着以后，搭建git server，在eclipse里集成git](http://simen-net.iteye.com/blog/832391)

3. [git常用命令](http://www.cnblogs.com/Jerry-Chou/archive/2012/05/14/2499088.html)