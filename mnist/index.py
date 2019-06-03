import tensorflow as tf
from keras.datasets import mnist
from keras.utils import np_utils

(x_train, y_train), (x_test, y_test) = mnist.load_data('mnist\mnist.npz')
# 60000张图片 60000个标签
print(x_train.shape, y_train.shape)
# 测试集 10000 10000
print(x_test.shape, y_test.shape)

# 观察数据
X_train = x_train.reshape(60000, 784)
X_test = x_test.reshape(10000, 784)
print(X_train.shape, type(X_train))
print(X_test.shape, type(y_test))

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train/=255
X_test/=255

# 观察标签

import numpy as np
import matplotlib.pyplot as plt
label,count=np.unique(y_train,return_counts=True)
print(label,count)

#one hot
n_classes=10
print(y_train.shape)
Y_train=np_utils.to_categorical(y_train,n_classes)
print(Y_train.shape)
Y_test=np_utils.to_categorical(y_test,n_classes)

