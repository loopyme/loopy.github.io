---
layout:     post
title: MaxCompute-DDL
subtitle: ACA课程学习笔记（3）
date:       2019-02-28
author:     Loopy
header-img: img/post-bg-alibaba.jpg
catalog: true
tags:
    - ACA-BigData
---

# A. 建表

``` sql
create table [if not exists] table_name
  [(col_name data_type [comment col_comment],...)]
  [comment table_comment]
  [partitioned by (col_name data_type [comment col_comment],...)]
  [lifecycle days]
  [as select_statement]
```

1. 大体和sql-ddl类似,其中有一点不同的是:分区列在分区声明位置(第三行)声明,而不应该出现表的字段定义(第二行)中

2. 建议: 不使用[if not exists].这是为了防止出现错误而不报错.所有操作都应确保操作者完全掌握数据库内容.所以删表时加上[if exists]是合适的.若在建表时出现重复定义(即表名重复):
 - 若使用了[if not exists],会跳过建表这一段继续向下执行,而不停止或报错.这容易使操作者误以为操作成功
 - 若未使用[if not exists],会停止执行,并报错

3. delete & rename
 - delete : ```drop table [if exists] table_name;```
 - rename : ```alter table table_name rename to new_table_name;```

4. 生命周期
 - 单位:天
 - 类型:正整数
 - 只能指定表,而不能是分区
 - 分区表与非分区表的
    - 非分区表:自最后一次数据被修改时刻开始计时
    - 分区表:分别判断每个分区是否应该被回收,但即使所有分区都被回收,表也不会被删除
 - 不指定就不回收

5. 快捷建表 CTAS
 - Like方法 : ```create table <table_name> as select <column_list> from <table_name> where...;```
 - As方法 :  ```create table <tanle_name> like <table_name>;```  

  区别:


  方法 | 带入数据  | 复制结构 | 带入LifeCycle |带入分区键信息,注释等|来源
  --- | --- | --- | --- |---|
  as  |✓|✓|||可以依赖多张表
  like||✓||✓|只能复制单张表的结构

# B. 分区
 - 添加分区 : ```alter table table_name add [if not exists] partition partition_spec;```
 - 删除分区```alter table table_name drop [if exists] partition_spec;```
 - 其中:```partition_spec:(partition_col1 = partition_col_value1,partition_col2 = partition_col_value2,...)```
 - 最高有六个不同的分区键

# C. 修改表属性
 - 添加列 : ```alter table table_name add columns (col_name type1, col_name2 type2);```
 - 改列名 : ```alter table table_name change column old_col_name rename to new_col_name;```
 - 表注释 : ```alter table table_name set comment 'tbl comment';```
 - 列注释 : ```alter table table_name change column col_name comment 'comment';```
 - 生命周期 : ```alter table table_name set lifecycle days;```
 - 修改时间 : ```alter table table_name touch [partition(partition_col='partition_col_value',...)];```

# D. 视图
  - 创建视图:
  ```sql
create [or replace] view [if not exists] view_name
  [(col_name[COMMENT col_comment],...)]
  [COMMENT view_comment]
  [AS select_statement]
  ```
  - 删除视图:```drop view[if not exists] view_name;```
  - 重命名视图:```alter view view_name rename to new_view_name;```
