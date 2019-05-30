#处理原始结构 csv
import pandas as pd
#展示图片
import seaborn as sns
#展示图片
import matplotlib.pyplot as plt
#展示图片
from mpl_toolkits import mplot3d
#矩阵处理
import numpy as np
import Utils.StatUtil as util

#2D的图
sns.set(context="notebook",style="whitegrid",palette="dark")
df0 = pd.read_csv("data0.csv",names=['square','price'])
sns.lmplot('square','price',df0,height=6,fit_reg=False)

print(df0.head())


#3D的图
df1=pd.read_csv('data1.csv',names=['square','bedrooms','price']);
df1.head()

fig=plt.figure()
ax=plt.axes(projection='3d')
ax.set_xlabel('square')
ax.set_ylabel('bedrooms')
ax.set_zlabel('price')
ax.scatter3D(df1['square'],df1['bedrooms'],df1['price'],c=df1['price'],cmap='Greens')

# 增加ones的图
df1=util.normalize_feature(df1);
ones=pd.DataFrame({'ones':np.ones(len(df1))})
df=pd.concat([ones,df1],axis=1);
print(df.head())

X_data=np.array(df[df.columns[0:3]])
Y_data=np.array(df[df.columns[-1]]).reshape(len(df),1)

print(X_data.shape,type(X_data))
print(Y_data.shape,type(Y_data))

import tensorflow as tf

alpha = 0.01
epoch=500
with tf.name_scope('input'):
    X=tf.placeholder(tf.float32,X_data.shape,name='X');
    Y=tf.placeholder(tf.float32,Y_data.shape,name='Y');

with tf.name_scope('hypothesis'):
    W=tf.get_variable("weights", (X_data.shape[1],1),initializer=tf.constant_initializer())
    y_pred=tf.matmul(X,W,name='y_pred')

with tf.name_scope('loss'):
    loss_op=1/(2*len(X_data))*tf.matmul((y_pred-Y),(y_pred-Y),transpose_a=True)
with tf.name_scope('train'):
    # 单轮训练操作
    opt=tf.train.GradientDescentOptimizer(learning_rate=alpha)
    train_op=opt.minimize(loss_op)

#运行数据流图

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    ##创建file writer实例
    writer=tf.summary.FileWriter("./summary/linear-regression-0/",sess.graph);
    for e in range(1,epoch+1):
        sess.run(train_op, feed_dict={X: X_data, Y: Y_data})
        if(e%10==0):
            loss,w=sess.run([loss_op,W],feed_dict={X:X_data,Y:Y_data})
            log_str="Epoch %d \t Loss=%.4g \t Model: y = %.4gx1 + %.4gx2 + %.4g"
            print(log_str % (e, loss, w[1], w[2], w[0]))
writer.close();