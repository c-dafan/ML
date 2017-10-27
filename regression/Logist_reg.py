# -*- coding: utf-8 -*-
"""
Created on Sun Sep 10 11:48:33 2017

@author: 大帆
"""
import numpy as np
from ..base.Baseclassifier import BaseClassifer


class Logistic_Reg(BaseClassifer):
    def __init__(self,lr=0.1,iter=200):
        self.lr=lr
        self.iter=iter

    def fit(self,X,y):
        shape=X.shape
        from sklearn.preprocessing import MinMaxScaler
        self.minmax=MinMaxScaler()
        X=self.minmax.fit_transform(X)
        X=np.c_[X,np.ones((shape[0],1))]
        shape=X.shape
        # self.W=np.zeros()
        self.W=np.zeros((1,shape[1]))
        for i in range(self.iter):
            dt=np.zeros([1,shape[1]])
            for j,(x_v,y_v) in enumerate(zip(X,y)):
#                if i>190 and j>97 :
#                    print(self._logist(np.dot(x_v,self.W.T)))
                dt+=(y_v-self._logist(np.dot(x_v,self.W.T)))*x_v
#            print(self.lr*dt/shape[0])
#            print(self.W)
#            print()
            self.W=self.W+self.lr*dt/shape[0]

    def _logist(self,x):
        # print(x)
        return 1/(1+np.exp(-x))
        
    def __one_zoro(self,x):
        y=np.zeros([x.shape[0],1])
        y[x>=0.5]=1
        y[x<0.5]=0
        return y
    def predict(self,X):
        shape=X.shape
        X=self.minmax.transform(X)
        X=np.c_[X,np.ones([shape[0],1])]        
        return self.__one_zoro(self._logist( np.dot(X,self.W.T)))

        
