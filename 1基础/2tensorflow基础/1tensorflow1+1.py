# -*- coding: utf-8 -*-
'''
@Author  : JackZhang9
@Time    : 2022/10/19 9:49
'''

import numpy as np
# 导入tensorflow
import tensorflow as tf


# 学习tensorflow前需要的前置知识
# python基本操作(赋值、分支及循环语句、使用import导入库)
# numpy库，python的一个常用科学计算库，tensorflow与之结合紧密
# 向量和矩阵运算(矩阵的加减法，矩阵与向量相乘，矩阵与矩阵相乘，矩阵的转置)
# 函数的导数，多元函数求导
# 线性回归
# 梯度下降法，求函数的局部最小值

# 可以将tensorflow视为一个科学计算库(类似于python下的numpy)
# tensorflow里只有一种数据就是tensor，与numpy里只有一种数据就是ndarray类似，基本很多numpy的操作tensorflow里可以平替

# 张量
# 定义一个张量，tf和numpy操作类似，

a0=tf.constant([[1,2,3],[4,5,6]])   # 自定义张量矩阵，和numpy里的numpy.array([])类似
print(a0)

# 和numpy相似，也能生成随机数据，可以符合gauss分布和uniform分布，还有截断gauss分布，只取2倍标准差里的数据，即发生概率是95%的数据
a=tf.random.uniform((),minval=1,maxval=50)  # 生成一个标量
print(a)

a2=tf.random.uniform([10],minval=5,maxval=60)  # 生成一个长度为10的向量
print(a2)

a3=tf.random.truncated_normal([3,3],mean=5,stddev=5)  # 生成一个3行3列的矩阵
print(a3)

a4=tf.random.normal([3,3],mean=5,stddev=5)
print(a4)


# 一些特殊全0或全1的张量
a5=tf.zeros([3,3])   # 全0矩阵
print(a5)

a6=tf.ones([3,3])  # 全1矩阵
print(a6)

a7=tf.eye(3,2)  # 第一个行数，第二个列数
print(a7)


# 张量属性，这一点也和numpy类似，通过相同的方法可以得到属性，如下
print(a7.shape,a7.dtype,a7.numpy)  # .numpy是将tensor转换为一个numpy的ndarray

# 张量里面元素指定，和numpy类似，numpy中是np.int32，在tensorflow中是tf.int32，都一样是通过dtype指定，
a8=tf.constant([2,2],dtype=tf.float32)
print(a8)


# 和numpy类似，tf中也有tensor的操作运算
# 如下：
b0=tf.random.normal([2,2])
b1=tf.random.uniform([2,2],minval=5,maxval=50)
b2=b0+b1  # 加法，和numpy类似
b21=tf.add(b0,b1)
print(b2,b21)

b3=b0@b1     # 乘法，和numpy类似，矩阵叉乘
b31=tf.matmul(b0,b1)
print(b3,b31)

b4=b0*b1  # 矩阵点乘
b41=tf.multiply(b0,b1)
print(b41,b4)














