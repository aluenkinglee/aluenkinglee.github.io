---
layout: post
title: "some commen list problems in the interview"
date: 2014-04-28 21:54
comments: true
categories: interview list 面试 链表
---

链表是最基本的数据结构，面试官也常常用链表来考察面试者的基本能力，而且链表相关的操作相对而言比较简单，也适合考察写代码的能力。链表的操作也离不开指针，指针又很容易导致出错。综合多方面的原因，链表题目在面试中占据着很重要的地位。本文对链表相关的面试题做了较为全面的整理，希望能对找工作的同学有所帮助。
链表结点声明如下：

<!-- more -->

```cpp
///=======链表
struct ListNode
{
    int data;
    ListNode * next;
};
```


#####  求单链表中结点的个数

最最基本的，应该能够迅速写出正确的代码，注意检查链表是否为空。时间复杂度为O（n）。参考代码如下：

```cpp
/// 求单链表中结点的个数
unsigned int getListLenght(ListNode* head)
{
    if(head == NULL)
        return 0;
    unsigned int len = 0;
    ListNode* current = head;
    for(; current != NULL; len++)
        current = current->next;
    return len;

}
```

##### 将单链表反转

从头到尾遍历原链表，每遍历一个结点，将其摘下放在新链表的最前端。注意链表为空和只有一个结点的情况。时间复杂度为O（n）。参考代码如下：

```cpp
/// 将单链表反转
ListNode * reverseList(ListNode * head)
{
    /// 边界
    if(head==NULL || head->next == NULL)
        return head;
    ListNode* cur = head;
    /// 反转后的新链表头指针，初始为NULL
    ListNode* rHead = NULL;
    ListNode* temp  = NULL;
    while(cur != NULL)
    {
        temp = cur;
        cur = cur->next;
        /// 将废弃结点摘下，插入新链表的最前端
        temp->next = rHead;
        rHead = temp;
    }
    return rHead;
}
```

##### 查找单链表中的倒数第K个结点（k > 0）

主要思路就是使用两个指针，先让前面的指针走到正向第k个结点，这样前后两个指针的距离差
是k - 1，之后前后两个指针一起向前走，前面的指针走到最后一个结点时，后面指针所指结点就是倒数第k个结点。

```cpp
///  查找单链表中的倒数第K个结点（k > 0）
ListNode * getRKthNode(ListNode * head, unsigned int k)
{
    if(k == 0 || head == NULL) /// 这里k的计数是从1开始的，若k为0或链表为空返回NULL
        return NULL;
    ListNode* ahead = head;
    ListNode* behind = head;

    for(; ahead!=NULL && k > 0 ; --k)   /// 遍历完前k个节点
        ahead = ahead->next;
    if( k > 0  )            /// 结点个数小于k，返回NULL
        return NULL;
    while(ahead != NULL)    /// 前后两个指针一起向前走，直到前面的指针指向NULL
    {
        ahead = ahead->next;
        behind = behind->next;
    }
    return behind;
}
```

##### 查找单链表的中间结点

也是设置两个指针，只不过这里是，两个指针同时向前走，前面的指针每次走两步，后面的指针每次走一步，前面的指针走到最后一个结点时，后面的指针所指结点就是中间结点，即第（n/2+1）个结点。注意链表为空，链表结点个数为1和2的情况。时间复杂度O（n）

```cpp
///  查找单链表的中间结点
///  获取单链表中间结点，若链表长度为n(n>0)，则返回第n/2+1个结点
ListNode * getMidNode(ListNode * head )
{
    /// 边界值处理，空或者只有一个节点
    if(head == NULL || head->next == NULL)
        return head;
    ListNode* ahead = head;
    ListNode* behind = head;

    /// 注意判断条件
    while(ahead != NULL)    /// 前面指针每次走两步，直到指向最后一个结点，后面指针每次走一步
    {
        ahead = ahead->next;
        behind = behind->next;
        /// 注意判断条件
        if(ahead != NULL)
            ahead = ahead->next;
    }

    return behind;
}
```

##### 从尾到头打印单链表

顺序颠倒，使用栈。系统栈或者自己写栈用。

```cpp
/// 逆序打印
void RPrintList2(ListNode* head)
{
    stack<ListNode*> s;
    ListNode * tmp = head;
    while(tmp!=NULL)
    {
        s.push(tmp);
        tmp = tmp->next;
    }
    while(!s.empty())
    {
        tmp = s.top();
        cout << tmp->data<<"\t";
        s.pop();
    }
    cout << endl;
}
void RPrintList3(ListNode* head)
{
    if(head == NULL)
        return;
    RPrintList3(head->next);
    cout <<head->data<<"\t";
}
```

##### 已知两个单链表pHead1 和pHead2 各自有序，把它们合并成一个链表依然有序

类似归并排序

```cpp
/// 已知两个单链表pHead1 和pHead2 各自有序，把它们合并成一个链表依然有序
/// 会改变原有链表结构，妥妥的
ListNode * MergeSortedList(ListNode * head1, ListNode * head2)
{
    if(head1 == NULL)
        return head2;
    if(head2 == NULL)
        return head1;
    ListNode* mergeHead = NULL;
    ListNode* temp = NULL;
    if(head1->data > head2->data)
    {
        mergeHead = head1;
        head1 = head1->next;
        mergeHead->next = NULL; /// 注意语句顺序，防止head1变成悬空节点
    }
    else
    {
        mergeHead = head2;
        head2 = head2->next;
        mergeHead->next = NULL; /// 注意语句顺序，防止head2变成悬空节点
    }
    temp = mergeHead;
    while(head1 != NULL && head2 != NULL)
    {
        if(head1->data>head2->data)
        {
            temp->next = head1;
            head1 = head1->next;
            temp=temp->next;
            temp->next=NULL;
        }
        else
        {
            temp->next = head2;
            head2 = head2->next;
            temp=temp->next;
            temp->next=NULL;
        }
    }
    if(head1!=NULL)
        temp->next = head1;
    if(head2!=NULL)
        temp->next = head2;
    return mergeHead;
}
```

递归版本

```cpp

/// 已知两个单链表pHead1 和pHead2 各自有序，把它们合并成一个链表依然有序
ListNode * MergeSortedList2(ListNode * head1, ListNode * head2)
{
    if(head1 == NULL)
        return head2;
    if(head2 == NULL)
        return head1;
    ListNode* mergeHead = NULL;
    if(head1->data > head2->data)
    {
        mergeHead = head1;
        head1 = head1->next;
        mergeHead->next = MergeSortedList2(head1,head2);
    }
    else
    {
        mergeHead = head2;
        head2 = head2->next;
        mergeHead->next = MergeSortedList2(head1,head2);
    }
    return mergeHead;
}
```

##### 判断两个单链表是否相交

如果两个链表相交于某一节点，那么在这个相交节点之后的所有节点都是两个链表所共有的。
也就是说，如果两个链表相交，那么最后一个节点肯定是共有的。先遍历第一个链表，记住
最后一个节点，然后遍历第二个链表，到最后一个节点时和第一个链表的最后一个节点做比
较，如果相同，则相交，否则不相交。时间复杂度为O(len1+len2)，因为只需要一个额外指
针保存最后一个节点地址，空间复杂度为O(1)

```cpp
bool isIntersected(ListNode * head1, ListNode * head2)
{
    if(head1 == NULL || head2 == NULL)
        return false;
    ListNode* tail1=head1;
    ListNode* tail2=head2;
    while(tail1 != NULL)
        tail1 = tail1->next;
    while(tail2 != NULL)
        tail2 = tail2->next;
    ///若相交，尾节点可定相同
    return (tail1 == tail2);
}
```

##### 求两个单链表相交的第一个节点

```cpp
ListNode* findFirstIntersectedNode(ListNode * head1, ListNode * head2)
{
    /// 边界值检查
    if(head1 == NULL || head2 == NULL)
        return NULL;
    ListNode* tail1=head1;
    ListNode* tail2=head2;
    int len1=0;
    int len2=0;
    while(tail1 != NULL)
    {
        tail1 = tail1->next;
        len1++;
    }
    while(tail2 != NULL)
    {
        tail2 = tail2->next;
        len2++;
    }

    ///若相交，尾节点肯定相同
    if(tail1 != tail2)
        return NULL;
    ListNode* node1=head1;
    ListNode* node2=head2;
    int k=0;
    /// 找到两个链表等长的位置。
    if(len1>len2)
    {
        k=len1-len2;
        while(k>0)
            node1=node1->next;
    }
    else
    {
        k=len2-len1;
        while(k>0)
            node2=node2->next;
    }
    /// 两个链表未相交之前，节点肯定不同
    while(node1!=node2)
    {
        node1=node1->next;
        node2=node2->next;
    }
    return node1;
}
```

##### 给出一单链表头指针 head 和一节点指针 node ，O(1)时间复杂度删除节点 node

对于删除节点，我们普通的思路就是让该节点的前一个节点指向该节点的下一个节点，这种情况需要遍历找到该节点的前一个节点，时间复杂度为O(n)。对于链表，链表中的每个节点结构都是一样的，所以我们可以把该节点的下一个节点的数据复制到该节点，然后删除下一个节点即可。要注意最后一个节点的情况，这个时候只能用常见的方法来操作，先找到前一个节点，但总体的平均时间复杂度还是O(1)

```cpp
void deleteNode(ListNode * head, ListNode * node)
{
    /// 边界值
    if(node==NULL)
        return ;
    if(node->next!=NULL)
    {
        /// 把后面的节点值复制到本位置
        node->data = node->next->data;
        ListNode* temp = node->next;
        node->next = node->next->next;
        delete temp;
    } else {
    /// 删除的是最后一个节点，只能用遍历法
        ListNode* cur = head;
        while( cur->next != node)
            cur = cur->next;
        cur->next=NULL;
        delete node;
    }
}
```
