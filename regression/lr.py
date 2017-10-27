import numpy as np
from ..base.Baseregress import Baseregree

class Linear_regress(Baseregree):
    def __init__(self,lr=0.01,iter=50):
        self.lr=lr
        self.birth=iter
    
    def fit(self,X,y):
        # X=np.arange()
        shape=X.shape
        from sklearn.preprocessing import MinMaxScaler
        self.minmax=MinMaxScaler()
        X=self.minmax.fit_transform(X)
        X=np.c_[X,np.ones([shape[0],1])]
        shape=X.shape
        self.W=np.zeros([1,shape[1]])
        for i in range(self.birth):
#            self.lr=self.lr-0.001
            dt=np.zeros([1,shape[1]])
            for x_v,y_v in zip(X,y):
                # print(x_v*self.W)
                dt=dt+(y_v-np.dot(x_v,self.W.T))*x_v
#            print(self.W)
#            print(self.lr*dt)
#            print()
            self.W=self.W+self.lr*dt/(shape[0])

    def predict(self,X):
        shape=X.shape
        X=self.minmax.transform(X)
        X=np.c_[X,np.ones([shape[0],1])]
        return np.dot(X,self.W.T)

    def __str__(self):
        return('lr:%f,birth:%f'%(self.lr,self.birth))



