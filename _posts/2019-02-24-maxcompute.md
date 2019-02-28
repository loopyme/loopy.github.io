---
layout:     post
title: MaxCompute
subtitle: MaxCompute平台架构整理
date:       2019-02-24
author:     Loopy
header-img: img/post-bg-alibaba.jpg
catalog: true
tags:
    - MaxCompute LearningNotes

---

>大数据计算服务（MaxCompute，原名ODPS）是阿里云的一种快速、完全托管的TB/PB级数据仓库解决方案.

阿里巴巴的业务其实就是在数加平台上的,而MaxCompute是数加核心的一个组件,我理解的是现在阿里不止想自己用,也想把数加云平台的概念推出来,让大家一起用,就像Hadoop生态一样.构建一个阿里的数加生态.特别的是,数加并不像Hadoop那样完全开源,毕竟是企业而不是社区开发维护的.

我整理出来的MaxCompute基础架构是这样的:
![MaxCompute](http://file.loopy.tech/pic/MaxCompute.png)
按照每一个模块的名字,应该就能知道关系,以及是怎么跑起来的了.计算层几乎可以被当作一个黑箱,由阿里开发和维护.这和Hadoop有点不一样.
