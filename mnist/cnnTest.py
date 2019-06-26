from keras.models import load_model
import tensorflow.gfile as gfile
import numpy as np
import tensorflow as tf
import os
from keras.datasets import mnist
from keras.utils import np_utils
from keras import backend as K
from keras.utils import np_utils
(x_train, y_train), (x_test, y_test) = mnist.load_data('mnist\mnist.npz')

img_rows, img_cols = 28, 28

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

X_train = x_train.astype('float32')
X_test = x_test.astype('float32')

X_train /= 255
X_test /= 255

n_classes = 10
Y_train = np_utils.to_categorical(y_train, n_classes)
Y_test = np_utils.to_categorical(y_test, n_classes)

print(x_train.shape, type(x_train))
print(x_test.shape, type(x_test))
save_dir="./mnist/model/keras_mnist.h5"
mnist_model=load_model(save_dir)

loss_and_metrics=mnist_model.evaluate(X_test,Y_test,verbose=2)

print("Test Loss:{}".format(loss_and_metrics[0]))
print("Test Accuracy: {}%".format(loss_and_metrics[1]*100))

predicted_classes=mnist_model.predict_classes(X_test)
correct_indices=np.nonzero(predicted_classes==y_test)[0]
incorrect_indices=np.nonzero(predicted_classes!=y_test)[0]

print("Classified correctly count: {}".format(len(correct_indices)))
print("Classified incorrectly count:{}".format(len(incorrect_indices)))