---
layout: post
title: "使用mongodb内置的mapreduce聚合函数统计词频"
date: 2015-01-14 13:51
comments: true
categories: mongodb map-reduce
---



有个需求需要对数据库中的某个字段进行频率统计，数据量较大，大概在2亿左右。

在之前先用一个简单的例子描述下问题。

比如我有一个mongo中的文档集合，比如说

```text
[
    { summary:"This is good" },
    { summary:"This is bad" },
    { summary:"Something that is neither good nor bad" }
]

>db.test.insert({ summary:"This is good" });
>db.test.insert({ summary:"This is bad" };
>db.test.insert({ summary:"Something that is neither good nor bad" });
```

现在要统计每个单词的出现次数，忽略大小写，最后按照升序排序，就像这样：

```text
[
    "is": 3,
    "bad": 2,
    "good": 2,
    "this": 2,
    "neither": 1,
    "nor": 1,
    "something": 1,
    "that": 1
]
```

所以我们可以使用mongo中自带的聚合函数框架——mapreduce，然后我们自己根据需求
写个map函数和reduce函数就可以了。

<!-- more -->

关于[Map-Reduce](http://docs.mongodb.org/manual/core/map-reduce/)可以看到更加
详细的介绍，尤其适用于服务器端的文档处理率操作。

然后看是map函数，每个doc会传给这个`map`函数，这个这个函数会查找`summary`这个字段，若存在，则变小写，根据空格分割字符串，每个字置为1.

```javascript
var map = function() {  
    var summary = this.summary;
    if (summary) { 
        // quick lowercase to normalize per your requirements
        summary = summary.toLowerCase().split(" "); 
        for (var i = summary.length - 1; i >= 0; i--) {
            // might want to remove punctuation, etc. here
            if (summary[i])  {      // make sure there's something
               emit(summary[i], 1); // store a 1 for each word
            }
        }
    }
};
```

然后在写`reduce`函数，它是把`map`函数得到的结果加起来，并最终返回每个字的结果。


```javascript
var reduce = function( key, values ) {    
    var count = 0;    
    values.forEach(function(v) {            
        count +=v;    
    });
    return count;
}
```

最后命令行执行

```bash
>db.test.mapReduce(map, reduce, {out: "word_count"})
>db.word_count.find().sort({value：-1})
```

即可看到结果，如下所示：

```text
{ "_id" : "is", "value" : 3 }
{ "_id" : "bad", "value" : 2 }
{ "_id" : "good", "value" : 2 }
{ "_id" : "this", "value" : 2 }
{ "_id" : "neither", "value" : 1 }
{ "_id" : "or", "value" : 1 }
{ "_id" : "something", "value" : 1 }
{ "_id" : "that", "value" : 1 }
```


### 统计重复字段

比如说要统计`taobao_id`这个字段，其他字段在这里用`summary`代指，这里面有的doc没有`taobao_id`字段，有的为空，这种情况会随软件版本升级出现类似情况。


```text
{ "summary" : "this is good", "taobao_id" : "ad23x@#sa" }
{ "summary" : "this is bad", "taobao_id" : "ad23x@#sa" }
{ "summary" : "this is holy shit!", "taobao_id" : "ad23x@#2323sa" }
{ "summary" : "you are so fucking clever", "taobao_id" : "ad123x@#2323sa" }
{ "summary" : "you r idiot", "taobao_id" : "" }
{ "summary" : "it is deliciouse" }
```


`map`函数如下:

```js
var map = function(){
    var taobao_id = this.taobao_id;
    // make sure there is the field.
    if(taobao_id)
    	// store a 1 for this taobao_id
        emit(taobao_id,1);
};
```

`reduce`函数如下：

```js
var reduce = function(key,values){
    var count = 0;
    values.forEach(function(v){
        count+=v;
    });
    return count;
};
```

结果如下：

```bash
> db.so.mapReduce(map,reduce,{out:"taobao_id_count"})
{
	"result" : "taobao_id_count",
	"timeMillis" : 58,
	"counts" : {
		"input" : 6,
		"emit" : 4,
		"reduce" : 1,
		"output" : 3
	},
	"ok" : 1,
}
> db.taobao_id_count.find()
{ "_id" : "ad123x@#2323sa", "value" : 1 }
{ "_id" : "ad23x@#2323sa", "value" : 1 }
{ "_id" : "ad23x@#sa", "value" : 2 }

```

在服务器上运行得到
如下结果

```text
 "result" : "taobao_id_count",
    "counts" : {
        "input" : NumberLong(166299476),
        "emit" : NumberLong(98911196),
        "reduce" : NumberLong(2492253),
        "output" : NumberLong(95269211)
    },
```

字段说明：
总共存在有1.6亿个doc（其实这个数还是不准，多谢俊德指出db.device.count()的个数为227186295）；

其中存在taobao_id的doc数为98911196个；

总共规约的个数（即taobao_id重复的个数大于1的）为2492253个；

最后的结果有95269211个。

没有taobao_id的doc数大约为7千万。

重复的taobao_id个数为使用如下命令得出，大约花费1分钟
db.taobao_id_count.count({"value":{$gt:1}})
2234285

结果验证：
db.taobao_id_count.find({"value":2})
得到一堆在重复个数为2的taobao_id集合。
在重复个数为2的taobao_id中，用它的taobao_id在device总集中查找，看返回结果是否为两个。db.device.find({"taobao_id":"+/quUacD9KYQAAcmQQglxvvO"})
结果只有两个结果


>>Referrence

1. [write scripts for the mongo shell](http://docs.mongodb.org/manual/tutorial/write-scripts-for-the-mongo-shell/)

2. [快速例子学习mongodb的mapreduce](http://jackyrong.iteye.com/blog/1408548)

3. [词频统计](http://stackoverflow.com/questions/16174591/mongo-count-the-number-of-word-occurences-in-a-set-of-documents)

4. [map reduce](http://docs.mongodb.org/manual/core/map-reduce/)

