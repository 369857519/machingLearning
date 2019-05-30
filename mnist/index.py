import tensorflow as tf
import matplotlib.pyplot as plt
minst=tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test)=minst.load_data('mnist\mnist.npz')
# 60000张图片 60000个标签
print(x_train.shape,y_train.shape)
# 测试集 10000 10000
print(x_test.shape,y_test.shape)

#观察数据
fig=plt.figure()
for i in range(15):
    plt.subplot(3,5,i+1)
    plt.tight_layout()
    plt.imshow(x_train[i],cmap='Greys')
    plt.title("Label:{}".format(y_train[i]))
    plt.xticks([])
    plt.yticks([])

