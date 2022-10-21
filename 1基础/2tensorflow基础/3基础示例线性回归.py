# -*- coding: utf-8 -*-
'''
@Author  : JackZhang9
@Time    : 2022/10/19 9:49
'''

import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import Input, Dense, Multiply
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical




#划分数据集
(x_train,y_train),(x_test,y_test) = mnist.load_data()
# x_train.shape


x_train = x_train.reshape(60000,784)
x_test = x_test.reshape(10000,784)


#设置数据类型为float32
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# 数据值映射在[0,1]之间
x_train = x_train/255
x_test = x_test/255

#数据标签one-hot处理
y_train = tf.keras.utils.to_categorical(y_train,10)
y_test = tf.keras.utils.to_categorical(y_test,10)
print(y_train[1])


def build_model():
    inputs = Input(shape=(input_dim,)) #输入层
    # ATTENTION PART STARTS HERE 注意力层
    attention_probs = Dense(input_dim, activation='softmax', name='attention_vec')(inputs)
    attention_mul =  Multiply()([inputs, attention_probs])
    # ATTENTION PART FINISHES HERE
    attention_mul = Dense(64)(attention_mul) #原始的全连接
    output = Dense(10, activation='relu')(attention_mul) #输出层
    model = Model(inputs=[inputs], outputs=output)
    return model


if __name__ == '__main__':
    m = build_model()  # 构造模型
    m.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
    m.summary()
    m.fit(x_train, y_train, epochs=20, batch_size=128)

    m.evaluate(x_test, y_test, batch_size=128)



















