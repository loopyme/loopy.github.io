---
layout:     post
title: Ubuntu18.04连接蓝牙鼠标修复
subtitle: 解决Ubuntu18.04下蓝牙连接不稳定的问题
date:       2019-03-22
author:     Loopy
header-img: img/post-bg-2015.jpg
catalog: true
tags:
    - ubuntu
---

## 问题描述

1. 软硬件环境
  - Controller: TninkPad-E470-BlueTooth4.1
  - Device: Microsoft Sculpt Comfort Mouse
  - OS: Ubuntu18.04

2. 问题复现情景
  - 蓝牙重启后不自动连接到鼠标,需要手动配对连接
  - 连接后暂停使用鼠标直至休眠(5min左右),即断开连接,无法自动重连
  - 只有鼠标出现异常,键盘,耳机均无异常

## 问题解决

1. 在终端里打开蓝牙控制器: ```bluetoothctl```
2. 显示可用列表,查看鼠标mac: ```devices```
3. Trust你的鼠标:(trust后面是你的鼠标mac)```trust 70:F0:87:23:FB:F4```
4. 要是还不行,就```sudo apt-get install pulseaudio-module-bluetooth```

搞定!(试了一万种方法,结果trust就够了...)

教训: 在网上的其他调整方法没穷尽之前,千万别去碰无辜的驱动!!!
