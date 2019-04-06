---
layout:     post
title: MaxCompute-DML
subtitle: ACA课程学习笔记（4）
date:       2019-03-07
author:     Loopy
header-img: img/post-bg-alibaba.jpg
catalog: true
tags:
    - ACA-BigData
---

# A. 查询 SELECT

``` sql
select [all|distinct] select_expr,select_expr,...
  from table_reference
  [where wher_condition]
  [group by col_list]
  [order by order_condition]
  [distribute by distribute_condition [sort by sort_condition]]
  [limit number]
```

注意:
- 列可以用列名指定,\*代表所有的列
- 支持嵌套子查询,但子查询必须要有别名(有了别名就和表差不多了)
- 子查询可以和其他表,或者互相之间join

# B. 更新 INSERT INTO/OVERWRITE

### 对普通表或静态分区:
``` sql
insert overwrite\into table tablename [partition(partcol1=val1,partcol2=val2,...)]
select_statement
from from_statement
```

### 对普通表或静态分区:
``` sql
insert overwrite table tablename partition (partcol1,partcol2,...)
select_statement from from_statement
```

注意:
 - 与sql不同,必须写表名
 - 不支持用value插一行数据
 - 知道具体插到哪个分区,用静态插;横跨多个分区,动态插
 - 如果目标表有多级分区,insert时允许部分静态,但静态分区必须是高级分区(高级分区指定插,低级分区动态插)
 - 动态生成的分区键不能为NULL

### 多路输出 MULTI INSERT:

从一个表读,写到多个表中.即"一读多写"

``` sql
from from_statement
insert overwrite\into table tablename1 [partition(partcol1=val1,partcol2=val2,...)]
  select_statement
[insert overwrite\into table tablename2 [partition...]
  select_statement2]
```

注意:
- 单个SQL最多写256路输出
- 对未分区表的表,或分区表的分区都不能作为目标出现多次
- 对同一张表的不同分区,不能同时overwrite和into

# C. 表关联 JOIN

关联方式:
- 左连接 left outer join: 返回LEFT
- 右连接 right outer join: 返回RIGHT
- 内连接 inner join: 返回交集
- 全连接 full outer join: 返回并集

``` sql
select [t1.col_name]
       [t2.col_name]
       ...
  from <tab_name1> t1 [left outer] join <tab_name2> t2 on <[t1.col_name =t2.col_name][and t1.col_name =t2.col_name]...>
```

注意:
 - 只允许and连接的等值条件
 - 最多支持16路join

## MAPJOIN
- 使用情景: 一个大表和一个或一群小表做join
- 基本原理: 把小表全丢内存里,加快join
- 注意事项:
  - 左(右)连接,左(右)表必须是大表
  - 内连接都可以做大表
  - 全连接不能用MAPJOIN
  - 小表可以为子查询
  - 引用小表或子查询,需要使用别名
  - 可以使用不等值连接或or
  - 最多制定6张小表
  - 小表占内存不超过2G

## 分支表达式 CASE WHEN
```sql
CASE <value>
  WHEN <condition_1> then <result_1>
  WHEN <condition_1> then <result_1>
  ...
  else <result_1>
END
```

```sql
CASE
  WHEN <condition_1> then <result_1>
  WHEN <condition_1> then <result_1>
  ...
  else <result_1>
END
```

注意:
 - 如果不一致,返回结果类型会变得一致
 - 非贪婪返回结果,碰到满足就返回
