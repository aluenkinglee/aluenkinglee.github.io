---
layout: post
title: "C++的hashtable那些事"
date: 2014-02-24 05:29
comments: true
categories: 
---

对于想知道hashtable原理的的人来说，多少对基本的数据结构和算法都有些了解，所以不再细说。

平常我们所说的一些容器来说，比如vector，list，stack之类，他们中的元素都是可以排序的，可以归为序列式容器，基于连续内存的vector可以实现随机存储，但是搜索的复杂度是O（n），list也是如此。
在关联式容器中，每个数据有一个键值和一个实值。元素插入到关联式容器中，容器内部结构根据其键值依据某个特定的规则将这个元素放置于适当的位置，从而实现插入和搜索是对数平均时间（基于红黑树），甚至是尽可能的是常数级别（就是这里说的散列表），其效率依赖于数据的特性和规则的设计。关联式的容器没有所谓的头尾，不会有begin(),end(),push_back()等这样的行为。

数组是个很好的内建数据结构，你甚至可以把数组看成一个最简单的hash表，index是种键值，而内容则是value。是不是很好，常数级寻址。但是也有问题，通常我们面对的键值不仅仅是简单的无符号的整数之流，还有很多的string啊，复杂的类对象啊，而这些问题就是hash函数所要处理的问题。hash函数把这个对象映射到[0-表的长度-1]之间。但是，不能够保证每个元素的键值与函数值是一一对应的，因此极有可能出现对于不同的元素，却计算出了相同的函数值，这样就产生了“冲突”，换句话说，就是把不同的元素分在了相同的“类”之中。 总的来说，“直接定址”与“解决冲突”是哈希表的两大特点。首先，为什么会出现冲突？基于这种想法设计出来的结构无疑是一种空间换时间的典范，尽管现在硬件越来越牛逼，但并不意味着可以随便挥霍，矛盾就出现在空间的数据往往小于我们要处理的数据规模，所以我们不可能设计出一种一一对应的函数来。而解决冲突就是hash第二个要解决的问题。

对于冲突的解决，往往有线性探测和二测探测以及开链法（类似list），前两种往往在数据规模小的情况下，即空间浪费的时候效率较高，这也意味着空间利用率不是很理想，而开链法，器负载系数大于1，就是索引的数组被极大利用了。
<!--more-->


所以在hashtable中，把一个元素（对象）插入到其中，分为以下的过程：

1.得到元素的键值Key。

2.调用处理该键型KeyType的hash函数(有时候需要用户自己编写)得到hash值（即下标）。

3.把该元素存放到对应该下标的桶内。

查找，取值的过程:

1.得到元素的键值Key。

2.调用处理该键型KeyType的hash函数得到hash值（即下标）。

3.比较桶的内部元素是否与key相等（编写equal_to函数，基本类型意外用户根据需求编写），若都不相等，则没有找到。

4.取出相等的记录的value。

综上所述，实现一个hashtable必须注意hash函数 和 比较函数的接口提供。

#####hashtable迭代器的设计

首先明确我们的需求，定位有hash函数搞定，而找到该桶之后，只是需要顺着桶往下找就够了，所以迭代器是个前向迭代器，对应的接口有实值，键值，hash仿函数，提取仿函数，等于仿函数，还有空间配置器等。其他编写仿照一般的迭代器设计即可。


```cpp

template <class Value, class Key,
         class HashFcn, class ExtractKey,
         class EqualKey, class Alloc>
struct __hashtable_iterator
{
    typedef hashtable<Value, Key, HashFcn, ExtractKey, EqualKey, Alloc> HashTable;
    typedef __hashtable_iterator<Value, Key,
            HashFcn, ExtractKey,
            EqualKey, Alloc>
            iterator;
    typedef __hashtable_iterator<const Value, Key,
            HashFcn, ExtractKey,
            EqualKey, Alloc>
            const_iterator;

    typedef __hashtable_nodes<Value> node;

    //因为是前向移动，没有后退操作--
    typedef forward_iterator_tag    iterator_categor;
    typedef Value                   value_type;
    typedef ptrdiff_t               difference_type;
    typedef size_t                  size_type;
    typedef Value*                  pointer;
    typedef Value&                  reference;

    node* cur;
    HashTable* ht;

    //constructor
    __hashtable_iterator(node* n, HashTable* tab) : cur(n), ht(tab) {}
    __hashtable_iterator() {}

    reference operator*()  const
    {
        return cur->val;
    }
    pointer   operator->() const
    {
        return &(operator*());
    }
    iterator& operator++();
    iterator  operator++(int);
    bool operator==(const iterator& it) const
    {
        return cur == it.cur;
    }
    bool operator!=(const iterator& it) const
    {
        return cur != it.cur;
    }
};
```

详情请看[__hashtable_iterator](https://github.com/aluenkinglee/stuff/blob/master/cplusplus/stl/hashtable#L127 "__hashtable_iterator")



#####hashtable的数据结构

由于采用开链的冲突解决方法，由此看来只需要一个vector代替桶，链表来装载桶内的元素即可。

需要注意的是hash函数，提取函数，等于函数的成员。另外需要buckets，还有元素数目的成员，有了这些基本就够了。


```cpp
//hashtable的数据结构
template <class Value, class Key,
         class HashFcn, class ExtractKey,
         class EqualKey, class Alloc>
class hashtable
{
public:
    typedef Value value_type;
    typedef value_type* pointer;
    typedef const pointer const_pointer;
    typedef value_type& reference;
    typedef const reference const_reference;
    typedef Key key_type;
    typedef HashFcn hasher;
    typedef EqualKey key_equal;
    typedef size_t size_type;
    typedef __hashtable_iterator<Value, Key, HashFcn, ExtractKey, EqualKey, Alloc> iterator;
    typedef typename __hashtable_iterator<Value, Key, HashFcn, ExtractKey, EqualKey, Alloc>::const_iterator const_iterator;
//constructor
public:
    hashtable(size_type n,
              const HashFcn &hf,
              const EqualKey &eql,
              const ExtractKey &ext)
        :hash(hf),equals(eql),get_key(ext),num_elements(0)
    {
        initialize_buckets(n);
    }

    hashtable(size_type n,
              const HashFcn &hf,
              const EqualKey &eql)
        :hash(hf),equals(eql),get_key(ExtractKey()),num_elements(0)
    {
        initialize_buckets(n);
    }

    hashtable(const hashtable &ht)
        :hash(ht.hash),equals(ht.equals),get_key(ht.get_key),num_elements(0)
    {
        copy_from(&ht);
    }

    ~hashtable()
    {
        clear();
    };

    hashtable& operator= (const hashtable &ht)
    {
        if (&ht != this)
        {
            clear();
            hash = ht.hash;
            equals = ht.equals;
            get_key = ht.get_key;
            copy_from(&ht);
        }
    }
private:
    hasher hash;
    key_equal equals;
    ExtractKey get_key;

    typedef __hashtable_nodes<Value> node;
    typedef simple_alloc<node, Alloc> node_allocator;

    //std::vector<node*, Alloc> buckets;
    std::vector<node*> buckets;
    size_type num_elements;
public:
    std::vector<node*> getBuckets() { return buckets;}
    //返回bucket vector大小
    size_type bucket_count() const
    {
        return buckets.size();
    }
    //返回bucket vector可能的最大值
    size_type max_bucket_count()
    {
        return __stl_prime_list[__stl_num_primes - 1];
    }
    //返回元素个数
    size_type size()
    {
        return num_elements;
    }
    //找到起始节点
    iterator begin()
    {
        size_type bucketIndex = 0;
        node* first;
        for (first = buckets[bucketIndex];
                !first && ++bucketIndex < buckets.size();
                first = buckets[bucketIndex]) {}
        return iterator(first, this);
    }
    //插入元素，不允许重复
    std::pair<iterator, bool> insert_unique(const Value& obj)
    {
        resize(num_elements + 1);
        return insert_unique_noresize(obj);
    }
    //插入元素，允许重复
    iterator insert_equal(const Value& obj)
    {
        resize(num_elements + 1);
        return insert_equal_noresize(obj);
    }
    //查找某一键值的节点
    iterator find(const Key& key)
    {
        size_type bucketIndex = bkt_num_key(key);
        node* first;
        for ( first = buckets[bucketIndex];
                first && !equals(get_key(first->val), key);
                first = first->next) {}
        return iterator(first, this);
    }
    //判断某一值出现的次数
    size_type count(const Key& key)
    {
        const size_type bucketIndex = bkt_num_key(key);
        size_type result = 0;
        for (const node* cur = buckets[bucketIndex];
                cur;
                cur = cur->next)
            if (equals(get_key(cur->val), key))
                ++result;
        return result;
    }
    //判断元素落在哪个bucket
    //提供两个版本
    //版本一：只接受实值
    size_type bkt_num(const Value& obj) const
    {
        return bkt_num_key(get_key(obj));
    }
    //版本二：接受实值和buckets个数
    size_type bkt_num(const Value& obj,size_type n) const
    {
        return bkt_num_key(get_key(obj),n);
    }
    //返回在index处的节点个数
    size_type elems_in_bucket(size_type bucketIndex)
    {
        size_type n = 0;
        node* tempNode = buckets[bucketIndex];
        while(tempNode && ++n) tempNode = tempNode->next;
        return n;
    }
    //整体删除
    void clear();
    //复制hash表
    void copy_from(const hashtable& ht);
private:
    //初始化buckets vector
    void initialize_buckets(size_type n)
    {
        const size_type n_buckets = next_size(n);
        buckets.reserve(n_buckets);
        buckets.insert(buckets.end(), n_buckets, (node*) 0);
        num_elements = 0;
    }
    //节点配置和释放函数
    node* new_node(const Value& obj)
    {
        //node *tempNode = node_allocator::allocate();
        node* tempNode = new node;
        tempNode->next = NULL;
        try
        {
            construct(&tempNode->val,obj);
        }
        catch (...)
        {
            //node_allocator::deallocate(tempNode);
            delete tempNode;
            return NULL;
        }
        return tempNode;
    }

    void delete_node(node *n)
    {
        destroy(&n->val);
        delete n;
    }

    //返回最接近n并大于等于n的质数
    size_type next_size(size_type n)const
    {
        return __get_next_prime(n);
    }

    //版本一：只接受键值
    size_type bkt_num_key(const Key& key) const
    {
        return hash(key) % (buckets.size());
    }
    //版本二：接受键值和buckets个数
    size_type bkt_num_key(const Key& key,size_type n) const
    {
        return hash(key) % n;
    }

    //判断是否需要扩充buckets vector，如有需要则进行扩充
    void resize(size_type num_elements_hint);
    //在不需要重新分配bucket vector的情况下插入元素，元素不允许重复
    std::pair<iterator, bool> insert_unique_noresize(const Value &obj);
    //在不需要重新分配bucket vector的情况下插入元素，元素不允许重复
    iterator insert_equal_noresize(const Value &obj);
};
```


至于更多的请看代码（由于是简易实现的，配置器那块并没有使用stl的配置器，简单写了下）[hashtable](https://github.com/aluenkinglee/stuff/blob/master/cplusplus/stl/hashtable#L127 "hashtable")


基于hashtable可以实现hash_set,hash_map,hash_multiset,hash_multimap,是这些容器的底层实现，而map,set,multiset,multimap则是基于rb-tree实现的，这是区别之一。

另外使用基于hashtable的时候需要提供hash仿函数，提取仿函数，等于仿函数这些参数。而基于rb-tree实现的需要比较函数即可。

平常时间根据查找速度, 数据量, 内存使用三个因素权衡，是否适合使用hashtable。