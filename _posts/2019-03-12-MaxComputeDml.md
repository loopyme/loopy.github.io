---
layout:     post
title: MaxCompute DML实验
subtitle: ACA课程学习笔记（6）
date:       2019-03-12
author:     Loopy
header-img: img/post-bg-alibaba.jpg
catalog: true
tags:
    - ACA-BigData
---

> 大数据计算服务（MaxCompute，原名 ODPS）是一种快速、完全托管的 GB/TB/PB 级数据仓库解决方案。MaxCompute 向用户提供了完善的数据导入方案以及多种经典的分布式计算模型，能够更快速的解决用户海量数据计算问题，有效降低企业成本，并保障数据安全。
>
> 内容: MaxCompute 只能以表的形式存储数据，并对外提供了 SQL 查询功能。用户可以将 MaxCompute 作为传统的数据库软件操作，但其却能处理TB、PB级别的海量数据。需要注意的是，MaxCompute SQL 不支持事务、索引及 Update/Delete 等操作，同时 MaxCompute 的 SQL 语法与 Oracle，MySQL 有一定差别，用户无法将其他数据库中的 SQL 语句无缝迁移到 MaxCompute 上来。此外，在使用方式上，MaxCompute SQL 最快可以在分钟，乃至秒级别完成查询，无法在毫秒级别返回用户结果。MaxCompute SQL 的优点是对用户的学习成本低，用户不需要了解复杂的分布式计算概念。具备数据库操作经验的用户可以快速熟悉 MaxCompute SQL 的使用。
>
>目标: MaxCompute SQL 采用的是类似于 SQL 的语法，可以看作是标准 SQL 的子集，但不能因此简单的把 MaxCompute 等价成一个数据库，它在很多方面并不具备数据库的特征，如事务、主键约束、索引等。本实验的目标是了解MaxCompute SQL 的DML语句（DML：Data Manipulation Language 数据操作语言），包括：SELECT查询、INSERT数据更新、多路输出、表关联JOIN、MAP JOIN、分支条件判断。

### 1 实验环境

1. 配置工具: 我是用的是[odpscmd命令行工具](http://repo.aliyun.com/odpscmd/?spm=a2c4g.11186623.2.17.2f9c5c23rsSTEm)操作的,也能通过浏览器从控制台打开

2. 登录实验环境: 使用申请到MaxCompute资源以后得到的AK ID 以及 AK Secret秘钥对 ,项目名称等信息配置在安装的客户端的配置文件(./conf/odps_config.ini)中,这时候在命令行就能打开odpscmd(./bin/odpscmd)了

### 2 建表准备数据

1. 构建实验表
  - 从本地上传sql:教程上给的```odpscmd –f /home/x/temp/dml_crt.sql```,但由于我没有安装odpscmd,所以也得把odpscmd的路径(./conf/odps_config.ini)加上去
  - ```show table;```: 检查表t_dml（一般表）,t_dml_p（分区表）是否成功创建

2. 加载数据:```tunnel upload /home/x/temp/t_dml.csv t_dml; ```教程上还加上了```-c GBK```,但由于我是Ubuntu的系统,加了反而会乱码

### 3 简单查询

在命令行工具里,可以直接回车换行,直到读到';'才会开始执行,执行后会print本次操作的ID,Log View等信息.

1. 一般查询
  - 检查表中“浙江省”相关的数据信息 ：```select * from t_dml where province='浙江省';```
  - 核查销售时间大于或等于某日期的数据信息： ```select city, amt from t_dml where sale_date >='2015-05-23 00:00:00';```
  - 检查总量大于某量的城市信息：```select distinct city from t_dml where amt > 700;```
  - 操作示例:


  ``` sh
  odps@ u_mygds0vm_1552387256>select distinct city from t_dml where amt > 800;

  ID = 20190312124442361g0tilssa
  Log view:
  /logview/?h=https://service.odps.aliyun.com/api&p=u_mygds0vm_1552387256&i=20190312124442361g0tilssa&token=SDRkOE1aWTBscGw0VkRRUUx5cWZDbUdUL2c4PSxPRFBTX09CTzpwNF8yMDA2Mzc1NTIzODcyNTU4NzMsMTU1Mjk5OTQ4Myx7IlN0YXRlbWVudCI6W3siQWN0aW9uIjpbIm9kcHM6UmVhZCJdLCJFZmZlY3QiOiJBbGxvdyIsIlJlc291cmNlIjpbImFjczpvZHBzOio6cHJvamVjdHMvdV9teWdkczB2bV8xNTUyMzg3MjU2L2luc3RhbmNlcy8yMDE5MDMxMjEyNDQ0MjM2MWcwdGlsc3NhIl19XSwiVmVyc2lvbiI6IjEifQ==
  Job Queueing.
  Summary:
  resource cost: cpu 0.02 Core * Min, memory 0.03 GB * Min
  inputs:
  	u_mygds0vm_1552387256.t_dml: 10001 (120448 bytes)
  outputs:
  Job run time: 1.000
  Job run mode: service job
  Job run engine: execution engine
  M1:
  	instance count: 1
  	run time: 0.000
  	instance time:
  		min: 0.000, max: 0.000, avg: 0.000
  	input records:
  		TableScan1: 10001  (min: 10001, max: 10001, avg: 10001)
  	output records:
  		StreamLineWrite1: 13  (min: 13, max: 13, avg: 13)
  R2_1:
  	instance count: 1
  	run time: 1.000
  	instance time:
  		min: 1.000, max: 1.000, avg: 1.000
  	input records:
  		StreamLineRead1: 13  (min: 13, max: 13, avg: 13)
  	output records:
  		AdhocSink1: 13  (min: 13, max: 13, avg: 13)


  +------+
  | city |
  +------+
  | 保康县 |
  | 古浪县 |
  | 天水市区城区 |
  | 无锡市 |
  | 武进县 |
  | 治多县 |
  | 济宁市 |
  | 深圳市 |
  | 罗平县 |
  | 胶南市 |
  | 苍山县 |
  | 阜宁县 |
  | 静安区 |
  +------+
```

2. 使用子句的查询
  - 统计浙江省销量大于某量的销售城市排名 ：
  ``` sh
  select city,sum(amt) as total_amt
    from t_dml
       where province='浙江省'
          group by city
            having count(*)>1  and sum(amt) > 2000
            order by total_amt desc
         limit 10;   
```
  - 城市排名统计
  ``` sh
  select city, cnt, amt
     from t_dml
     distribute by city
  sort by cnt;
```

### 4 数据更新
1. 追加记录
  - 从dual往t_dml里插数据:
  ``` sh
  insert into table t_dml select -1,'1900-01-01 00:00:00','','',0,0,0 from dual;
  ```
  - 检查结果:
  ``` sh
  select * from t_dml where detail_id=-1;
  ```

2. 分区表数据操作
  - 添加分区:
  ``` sh
  alter table t_dml_p add if not exists partition (sale_date='2015-01-01');
  ```
  - 往分区里添加数据:
  ```sh
  insert into table t_dml_p partition (sale_date='2015-01-01')
  select -1, '', '', 0, 0, 0 from dual;
  ```
  - 检查结果:
  ```sh
  select * from t_dml where detail_id=-1;
  ```

### 5.
