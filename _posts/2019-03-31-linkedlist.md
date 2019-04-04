---
layout:     post
title: 使用链表计算一元稀疏多项式加法
subtitle: 数据结构作业
date:       2019-03-31
author:     Loopy
header-img: img/post-bg-2015.jpg
catalog: true
tags:
    - Algorithm
---

> 按照“作业截至即公开”的策略，我决定使用数据结构的弱智作业来水一篇日志，并延时到作业截至后再post上来，实际上，这是2019-03-28写的了。

> 题目；
> 1. 编写使用freelist 的带头、尾结点的双向链表类的定义，实现双向链表的基本操作。
> 2. 利用双向链表实现2个一元稀疏多项式的加法运算，运算结果得到的链表要求按照指数升序有序，并遍历输出指数升序、指数降序的多项式。

### 大致思路

就是创造一个链表类的子类，节点数据能表示一个项的稀疏和阶数。然后画出来多项式类的协作图，差不多是这样：
![multinomial.png](http://api.loopy.tech/api/multinomial/class_multinomial__coll__graph.png)

然后弱智的写就行了。

### 1. 链表类

从书上抄，再自己加个构造函数就行了。可惜我最终还是没找到针对代码的OCR，只能手打。
```c++
/// 双向链表的节点

template<typename E>
class Link {
private:
    static Link<E> *freelist;
public:
    E element;
    Link *next;
    Link *prev;

    explicit Link(const E &it, Link *prevp, Link *nextp) {
        element = it;
        prev = prevp;
        next = nextp;
    }

    explicit Link(Link *prevp = nullptr, Link *nextp = nullptr) {
        prev = prevp;
        next = nextp;
    }

    void *operator new(size_t) {
        if (freelist == NULL) return ::new Link;
        Link<E> *temp = freelist;
        freelist = freelist->next;
        return temp;
    }

    void operator delete(void *ptr) {
        ((Link<E> *) ptr)->next = freelist;
        freelist = (Link<E> *) ptr;
    }

};

template<typename E>
Link<E> *Link<E>::freelist = NULL;
```


```c++
/// 双项链表

template<typename E>
class LinkedList {
protected:
    Link<E> *curr;
    Link<E> *head;
    Link<E> *tail;
    int cnt;
public:
    LinkedList() {
        curr = head = new Link<E>;
        curr->next = tail = new Link<E>;
        tail->prev = curr;
        cnt = 0;
    }

    void insert(const E &it) {
        curr->next = curr->next->prev = new Link<E>(it, curr, curr->next);
        cnt++;
    }

    void append(const E &it) {
        tail->prev = tail->prev->next = new Link<E>(it, tail->prev, tail);
        cnt++;
    }

    E remove() {
        if (curr->next == tail) {
            return 0;
        }
        E it = curr->next->element;
        Link<E> *ltemp = curr->next;
        curr->next->next->prev = curr;
        curr->next = curr->next->next;
        delete ltemp;
        cnt--;
        return it;
    }

    void prev() {
        if (curr != head) {
            curr = curr->prev;
        }
    }
};

```

### 2. 项的结构体

这里有个小东西，因为助教题也不说清楚，所以只有依靠Define预留参数，到时候针对OJ再调整参数，控制编译。

> *上完实验课更新：*
> 总结出了垃圾OJ必须具有的品质：
> 1. 登不上（用户数据库炸了
> 2. 登上了没有题（题目数据库炸了
> 3. 提交了全部打零分（测试用例是错的或者没有
> 4. 代码全部无法保存（设计者牛逼！
> 5. 不计算使用内存，时间（我怀疑根本就没有测试容器
> 6. 好不容易能跑了，一共就十个测试用例，还没有特殊点的（出题者牛逼！

```c++
///////////////////////////// 测试用例参数(写的时候还没公布测试用例..)

// 多项式系数&阶数的数据类型

# define number int

/////////////////////////////

/// 存储每一项的系数和阶数

struct Term {
    number coefficient;
    number order;
};

```
### 3. 多项式类

别看它长，写了30min的成果，大多代码段是相似的，直接复制再改，都是树形的判断条件导致的。

``` c++
///////////////////////////// 测试用例参数(写的时候还没公布测试用例..)

// 输入的多项式阶数是否有序

//# define isSorted

// 输入的多项式阶数是否按降序

//# define isDescending

// 系数为0或次数为0是否需要输出

// define printzero

/////////////////////////////

/// 多项式类,继承自LinkedList

class Multinomial : public LinkedList<Term> {
public:
    /**
  * @brief 多项式的构造函数,从输入流里读取
  * @param termsCount 项的个数
  */
    Multinomial(int termsCount) {
        int i;
        number coefficientBuffer, orderBuffer;
        for (i = 0; i < termsCount; i++) {
            cin >> coefficientBuffer >> orderBuffer;
            append(Term() = {coefficientBuffer, orderBuffer});
        }

#ifndef isSorted

        // bubble sort to get ascending

        curr = head->next;
        Link<Term> *j;
        Term buffer;

        while (curr != tail) {
            j = curr->next;
            while (j != tail) {
                if (curr->element.order > j->element.order) {
                    buffer = curr->element;
                    curr->element = j->element;
                    j->element = buffer;
                }
                j = j->next;
            }
            curr = curr->next;
        }

#endif

#ifdef isDescending

        Link<Term> *buffer;
        curr = head;
        while (curr != tail) {
            buffer = curr->next;
            curr->next = curr->prev;
            curr->prev = buffer;
            curr = curr->prev;
        }
        buffer = head;
        head = tail;
        tail = buffer;
#endif
    }

    /**

* @brief 多项式的输出

* @param isAscending 布尔,是否升序输出

* @todo Overload cout

*/
    void print(bool isAscending = true) {

#ifdef printzero

        if (isAscending) {
            curr = head->next;
            while (true) {
                cout << curr->element.coefficient << "x^" << curr->element.order;
                curr = curr->next;
                if (curr == tail) {
                    cout << endl;
                    break;
                } else {
                    cout << "+";
                }
            }
        } else {
            curr = tail->prev;
            while (true) {
                cout << curr->element.coefficient << "x^" << curr->element.order;
                curr = curr->prev;
                if (curr == head) {
                    cout << endl;
                    break;
                } else {
                    cout << "+";
                }
            }
        }

#endif

#ifndef printzero

        bool isFirstTerm = true;
        if (isAscending) {
            curr = head->next;
            while (curr != tail) {
                if (curr->element.coefficient == 0) {
                } else if (curr->element.order == 0) {
                    if (isFirstTerm) {
                        cout << 1;
                        isFirstTerm = false;
                    } else {
                        cout << "+1";
                    }

                } else {
                    if (isFirstTerm) {
                        cout << curr->element.coefficient << "x^" << curr->element.order;
                        isFirstTerm = false;
                    } else {
                        cout << "+" << curr->element.coefficient << "x^" << curr->element.order;
                    }
                }
                curr = curr->next;
            }
            cout << endl;
        } else {
            curr = tail->prev;
            while (curr != head) {
                if (curr->element.coefficient == 0) {
                } else if (curr->element.order == 0) {
                    if (isFirstTerm) {
                        cout << 1;
                        isFirstTerm = false;
                    } else {
                        cout << "+1";
                    }
                } else {
                    if (isFirstTerm) {
                        cout << curr->element.coefficient << "x^" << curr->element.order;
                        isFirstTerm = false;
                    } else {
                        cout << "+" << curr->element.coefficient << "x^" << curr->element.order;
                    }
                }

                curr = curr->prev;
            }
            cout << endl;
        }
#endif
    }
/**

* @brief 多项式加法,结果直接存在it里

* @param m 加法的加数

* @todo Overload +

*/

    void add(Multinomial m) {
        curr = head->next;
        m.curr = m.head->next;
        while (true) {
            if (curr->element.order > m.curr->element.order) {
                curr = curr->prev;
                insert(m.curr->element);
                curr = curr->next;
                m.curr = m.curr->next;
            } else if (curr->element.order == m.curr->element.order) {
                curr->element.coefficient += m.curr->element.coefficient;
                curr = curr->next;
                m.curr = m.curr->next;
            } else if (curr->element.order < m.curr->element.order) {
                curr = curr->next;
            }

            if (curr == tail) {
                while (m.curr != m.tail) {
                    append(m.curr->element);
                    m.curr = m.curr->next;
                }
                break;
            } else if (m.curr == m.tail) {
                break;
            }

        }

    }
};
```

### 4.文档
以前写了个shell脚本，对CPP项目Git+Doxygen一键操作，还能自动上传到html服务器，于是就有了[这个东西](http://api.loopy.tech/api/multinomial/index.html)。

所以这个项目的自动头部注释长这样（别去clone那个仓库...那是个私库）
```
/**
 * @File    : Multinomial.h
 * @Desc    : 链表,链表节点,多项式,项,三个类加一个结构体
 * @Date    : 19-3-28 下午6:53
 * @Author  : Loopy
 * @Git     : https://github.com/loopyme/Stuffs.git
 * @ApiDoc  : http://api.loopy.tech/api/multinomial/index.html
 * @Contact : 57658689098a@gmail.com
 * @Software: CLion
 */
```

### 搞定
> *上完实验课更新：*
>
> 等到什么时候有空，我一定写一篇“如何在一下午从0开始配置好一个OJ平台”。。。
