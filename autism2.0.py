import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder 
import random as rd
 
# label_encoder = preprocessing.LabelEncoder() 
dataset=pd.read_csv("Cleaned_Autism_Data.csv",converters={"Score":int})
X = dataset.iloc[:,[4,7]].values
le = LabelEncoder()
X[:,1] = le.fit_transform(X[:,1])
m=X.shape[0]
n=X.shape[1]
n_iter=100
K=2
Centroids=np.array([]).reshape(n,0)
for i in range(K):
    rand=rd.randint(0,m-1)
    Centroids=np.c_[Centroids,X[rand]]

Output={}

for i in range(n_iter):
     
      EuclidianDistance=np.array([]).reshape(m,0)
      for k in range(K):
          tempDist=np.sum((X-Centroids[:,k])**2,axis=1)
          EuclidianDistance=np.c_[EuclidianDistance,tempDist]
      C=np.argmin(EuclidianDistance,axis=1)+1
     
      Y={}
      for k in range(K):
          Y[k+1]=np.array([]).reshape(2,0)
      for i in range(m):
          Y[C[i]]=np.c_[Y[C[i]],X[i]]
     
      for k in range(K):
          Y[k+1]=Y[k+1].T
    
      for k in range(K):
          Centroids[:,k]=np.mean(Y[k+1],axis=0)
      Output=Y


plt.scatter(X[:,0],X[:,1],c='black',label='unclustered data')
plt.xlabel('Score')
plt.ylabel('Autism')
plt.legend()
plt.title('Plot of data points')
plt.show()
color=['red','blue']
labels=['cluster1','cluster2']
for k in range(K):
    plt.scatter(Output[k+1][:,0],Output[k+1][:,1],c=color[k],label=labels[k])
plt.scatter(Centroids[0,:],Centroids[1,:],s=300,c='yellow',label='Centroids')
plt.legend()
plt.show()




