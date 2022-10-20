# -*- coding: utf-8 -*-
'''
@Author  : JackZhang9
@Time    : 2022/10/19 9:49
'''

import tensorflow as tf
#忽略
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 在机器学习中，需要计算函数的导数，
# 在深度学习中，tensorflow提供了自动求导机制，
# 在即时执行模式下，tensorflow使用tf.GradientTape()这个求导记录器实现自动求导。

# 如下：
a=tf.Variable(5.)  # 初始化一个变量,默认能够被自动求导
# 打开一个求导记录器，在GradientTape()内的上下文，所有计算步骤都会倍记录用于求导
with tf.GradientTape() as tape:
    b=tf.square(a)
y_grad=tape.gradient(b,a)
print(b)
print(y_grad)









































