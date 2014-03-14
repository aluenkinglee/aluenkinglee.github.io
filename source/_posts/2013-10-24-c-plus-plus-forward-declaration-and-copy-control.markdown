---
layout: post
title: "C++前置声明和复制控制"
date: 2013-10-24 05:33
comments: true
categories: C++ predeclaration  赋值  复制 复制控制
---
突然有兴致想起了看会C++，因为最近一直是python，java，python的节奏...在这样下去，C++水平恐怕就停留在了只会编算法的地儿了...

随手一翻，看到了复制控制。对于这章，印象里的记忆是：
>> 如果一个类它有形如指针或者申请了其他的系统资源成员，这个时候就得注意了，如申请资源，如何释放资源，复制的时候应该注意这些成员的行为是怎么样的。

然后，就不是很清楚了。大概看了看 其实也差不多...好吧是差很多。编程的过程中出现了一些问题。在这里记录下来>.< 真实忘不了了！

#####复制构造函数
它是一个特殊的构造函数，而且形参常用const Type& 来修饰(如果凶残点，用指针也不是不行，但一定得是这两种！想想是为什么？)

<!--more-->
有两种情况会调用它：

1. 定义一个新对象，并用一个同类型的对象对它初始化，就像这样 `string fileDesp(filename);`此处的filename是已经定义好的对象。'显示调用'
2. 复制一个对象，并把它作为实参传给一个函数。'隐式调用'
3. 从函数返回时复制一个对象。'隐式调用'
4. 初始化顺序容器中的元素。'隐式调用'
5. 根据元素初始化列表初始化数组元素。'隐式调用'

```cpp
string book = "ISBN-2323-23234";
```

创建book对象时，编译器首先会接受一个C语言风格形参的string构造函数，创建一个临时对象，然后调用string的复制构造函数将book初始化那个临时对象的副本。（但是我感觉更像是'临时对象->赋值操作'..这里需要编码测试一下）

和java，python创建对象的方法来比,C++方法真心多...

#####合成的复制构造函数
就是自个没写编译器默认提供的复制构造函数，完成的功能很简单，数据成员逐个初始化，（`static`成员例外！！因为他们是属于类的！！）

#####关于explicit的复制构造函数
(以后加上，真心没写过，只知道IO类型的复制构造函数都是explicit的)

#####赋值重载
=是个二元运算符，所以有两个形参，分别对应左操作数和右操作数(const引用),当为成员函数时就是左操作数默认绑定到this指针上了。返回值为同一类型的引用。

#####关于析构函数
析构函数就是用来回收那些申请的系统资源的。所以自个度量何时该自己写给类的析构函数吧。

#####关于何时调用析构函数
1. 撤销类的对象自动调用
2. 动态分配的对象只有在删除指向该指针时，才会运行析构函数！！否则会导致内存泄漏，小心啦。

```cpp
string filename * p = new string();
delete p;
```
上述删除的行为不会销毁属于类的成员对象——static成员！其实挺好理解的啊。


下面是个例子。现在先在这里贴下代码，以后放到github里面去。


```cpp
#include <iostream>
#include <set>

using std::string;
using std::set;
using std::cout;
using std::endl;
//predeclaration of Message
class Message;

class Folder
{
public:
    Folder(){
    	cout << "Folder" <<endl;
    }
    Folder(const Folder&);
    Folder& operator=(const Folder &);
    ~Folder();

    void save(Message&);
    void remove(Message&);
    void addMsg(Message*);
    void remMsg(Message*);

private:
    set<Message*> messages;

    void put_Folder_in_Message(const set<Message*>&);
    void remove_Folder_from_Message();

};

class Message
{
public:
    Message(const string & str=""):
        contents(str)
        {}
    Message(const Message&);
    Message& operator=(const Message &);
    ~Message();

    void save(Folder&);
    void remove(Folder&);
    void addFldr(Folder*);
    void remFldr(Folder*);

private:
    string contents;
    set<Folder*> folders;

    void put_Msg_in_Folders(const set<Folder*>&);
    void remove_Msg_from_Folders();
};


Folder::Folder(const Folder& f):
    messages(f.messages)
{
    put_Folder_in_Message(messages);
}

void Folder::put_Folder_in_Message(const set<Message*> & msg)
{
    for (set<Message*>::const_iterator beg = msg.begin();
            beg != msg.end();
            ++beg)
    {
        (*beg)->addFldr(this);
    }
}

Folder& Folder::operator=(const Folder & f)
{
    if (&f != this)
    {
        //先把自己的给清除掉，在加上要赋值的，要不然肯定不一样。。
        //因为Messages不一样。。
        remove_Folder_from_Message();
        messages = f.messages;
        put_Folder_in_Message(messages);
    }
    return *this;
}

void Folder::remove_Folder_from_Message()
{
    for(set<Message*>::const_iterator beg = this->messages.begin();
            beg != messages.end();
            ++beg)
    {
        (*beg)->remFldr(this);
    }
}

Folder::~Folder()
{
    remove_Folder_from_Message();
}

void Folder::save(Message & msg)
{
    addMsg(&msg);
    msg.addFldr(this);
}

void Folder::remove(Message& msg)
{
    remMsg(&msg);
    msg.remFldr(this);
}

void Folder::addMsg(Message* msg)
{
    messages.insert(msg);
}

void Folder::remMsg(Message* msg)
{
    messages.erase(msg);
}

//copy construction ,put the new message into the folders where the msg is pointed.
inline Message::Message(const Message& m):
    contents(m.contents), folders(m.folders)
{
    put_Msg_in_Folders(folders);
}

void Message::put_Msg_in_Folders(const set<Folder*> &folders)
{
    for(set<Folder*>::const_iterator beg = folders.begin();
            beg != folders.end();
            ++ beg)
    {
        //beg is a pointer to Folder*
        (*beg)->addMsg(this);
    }
}

inline Message& Message::operator=(const Message& msg)
{
    if(&msg != this) {
        //首先把自己指向的那些folder都给取消掉
        remove_Msg_from_Folders();
        //消息的内容copy过来
        contents = msg.contents;
        folders = msg.folders;
        put_Msg_in_Folders(folders);
    }
    return *this;
}

void Message::remove_Msg_from_Folders()
{
    for(set<Folder*>::const_iterator beg = folders.begin();
            beg != folders.end();
            ++beg)
    {
        (*beg)->remMsg(this);
    }
}

inline Message::~Message()
{
    remove_Msg_from_Folders();
}

void Message::save(Folder& folder)
{
    addFldr(&folder);
    folder.addMsg(this);
}

void Message::addFldr(Folder* pfolder)
{
    folders.insert(pfolder);
}

void Message::remove(Folder& folder)
{
    remFldr(&folder);
    folder.remMsg(this);
}

void Message::remFldr(Folder* pfolder)
{
    folders.erase(pfolder);
}

int main()
{
	Message m("dsfasdf");
	Folder f = Folder();
	cout << "asdf"<< endl;
	return 0;
}

```

可以运行，但是不是想要的。

正确的应该分开写

`Folder.h`

```cpp
#include <iostream>
#include <set>

#ifndef __Folder__
#define __Folder__

using std::string;
using std::set;


#include "Message.h"
//predeclaration of Message
class Message;
// Message is a incomplete type.It can be used in limited ways only.
// 1.can not define object of this type.
// 2.only used as a pointer or ref.
// 3.declare it as the formal parameter of a function or return type of a function.
class Folder
{
public:
    Folder(){}
    Folder(const Folder&);
    Folder& operator=(const Folder&);
    ~Folder();

    // Message is used as the formal parameter.
    void save(Message&);
    void remove(Message&);
    void addMsg(Message*);
    void remMsg(Message*);

private:
    // Message is used as the typename of the template.
    set<Message*> messages;

    void put_Folder_in_Message(const set<Message*>&);
    void remove_Folder_from_Message();

};

#endif
```

`Message.h`

```
#include <iostream>
#include <set>

#ifndef __Message__
#define __Message__

using std::string;
using std::set;

#include "Folder.h"
//predeclaration of Folder
class Folder;
// Folder is a incomplete type.It only can be used in limited ways.
// 1.can not define object of this type.
// 2.only used as a pointer or ref.
// 3.declare it as the formal parameter of a function or return type of a function.

class Message
{
public:
    Message(const string & str=""):
        contents(str) {}
    Message(const Message&);
    Message& operator=(const Message &);
    ~Message();

    // Folder is used as the formal parameter.
    void save(Folder&);
    void remove(Folder&);
    void addFldr(Folder*);
    void remFldr(Folder*);

private:
    string contents;
    set<Folder*> folders;

    // Folder is used as the typename of the template.
    void put_Msg_in_Folders(const set<Folder*>&);
    void remove_Msg_from_Folders();
};

#endif
```

一个简单的例子，就是来测试一下`class A=B`时究竟会不会同时调用赋值构造函数和拷贝构造函数。

```cpp
#include <iostream>
using namespace std;

class object
{
private:
    int data;
public:
    object(int d = 0):
        data(d)
    {
        cout << "default constructor" << endl;
    }
    
    object(const object& other):
        data(other.data)
    {
        cout << "copy constructor" <<endl;
    }
    
    object& operator=(const object& other)
    {
        if(&other != this)
        {
            data = other.data;
            cout << "assignment constructor" <<endl;
        }
        return *this;
    }
};

void behavior1(object other)	//形参调用copy constructor
{
    cout << "test behavior1" << endl;
}

void behavior2(object& other)
{
    cout << "test behavior2" << endl;
}
int main()
{
    object A;
    cout << endl;
    object B(A);
    cout << endl;
    object C = A;  //这个情况仍然只是调用copy constructor
    cout << endl;
    C=B;	   //只有这种情况下才会调用assignment constructor
    cout << endl;
    behavior1(A);
    cout << endl;
    behavior2(A);
    return 0;
}
```

测试结果显示如下：

```text
default constructor

copy constructor

copy constructor

assignment constructor

copy constructor
test behavior1

test behavior2
```