import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

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


