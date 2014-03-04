---
layout: post
title: "在不同PC上协同写作同一个Octopress博客"
date: 2013-11-13 15:20
comments: true
categories: Octopress 协同写作 co-write
---

因为需要在不同的地方写博客，在加上之前错误的操作，所以有了这篇文章。

假设已经学会了然后安装Octopress博客。

## Octopress 分支说明

Octopress的git仓库(repository)有两个分支，分别是`master`和`source`，其中：

1. `master`存储的是博客网站本身，github基于此对页面渲染。该分支根目录处在`_deploy`文件夹，由`rake deploy`命令推送到服务器，一般而言，我们不需要对`master`做任何操作。

2. `source`存储的是生成博客的源文件（各种markdown文件）,写作博客是在这个分支。每次写完之后，记得推送到服务器。这样就不用担心我们的备份了。

<!--more-->

## 克隆服务器上的git到新机器

将博客的源文件clone到本地的（假设就叫做）octopress文件夹内。

```
$ git clone -b source git@github.com:username/username.github.com.git octopress
```
接下来这步骤是最重要的，当初栽在这儿了。(会出现`No such file or directory - _deploy`)

```
$ cd octopress
$ git clone -b master git@github.com:username/username.github.com.git _deploy 
```

还是要安装博客的。

```
$ gem install bundler
$ rbenv rehash    		# If you use rbenv, rehash to be able to run the bundle command
$ bundle install
$ rake setup_github_pages	#执行初始化
```

然后会提示输入仓库的ssh url。

```
Enter the read/write url for your repository
(For example, 'git@github.com:your_username/your_username.github.com)
```

## Co-write 协同写作博客

如果你是和别人合作博客，或者自己同时在好几个电脑上写博客，每次开始之前，git pull origin source获得最新的文件,rake generate生成新的页面.更新master并不是必须的，因为你更改源文件之后还是需要rake generate的.

```
$ cd octopress
$ git pull origin source  	# update the local source branch
$ cd ./_deploy
$ git pull origin master  	# update the local master branch
```

在source分支做了博客的发布，或者改变了博客的设置之后

```
$ rake generate		 	#生成新的页面

$ git add .
$ git commit -am "Some comment here." 
$ git push origin source  	# 上面三行是更新远端source分支

$ rake deploy             	# 更新远端master分支，文章就发布到了博客中
```

## Reference

1. [Octopress - 像黑客一样写博客](http://williamherry.com/blog/2012/07/20/octopress-setup/)

2. [Setting up a Blog and Contributing to an Existing One](http://code.dblock.org/octopress-setting-up-a-blog-and-contributing-to-an-existing-one)

3. [在多台电脑上写Octopress博客](http://boboshone.com/blog/2013/06/05/write-octopress-blog-on-multiple-machines/)

4. [Octopress: 新手教程](http://www.whispering.co/blog/2011/12/03/octopress-for-freshman/)