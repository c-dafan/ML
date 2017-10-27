# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 16:06:16 2017

@author: 大帆
"""

import numpy as np
from sklearn.datasets import load_boston

data=load_boston()
X=data.data
y=data.target

def kmeans(X,k):
    shape=X.shape
    cen=np.arange(X.shape[0])
    np.random.shuffle(cen)
    cen=cen[:k]
#    cen=np.random.randint(0,shape[0],size=k)
    cen=X[cen]
    for j in range(100):
        least_cen=cen
        cen=np.tile(cen,[shape[0],1,1])
        cen=np.transpose(cen,[1,0,2])
        juli=cen-X
        juli=juli**2
        juli=juli.sum(axis=2)
        a=juli.argmin(axis=0)
        cen=[]
        for i in range(k):
            cen.append(X[a==i].mean(axis=0))
        
        cen=np.array(cen)
        if (least_cen!=cen).sum()==0:
            return [a,cen]
    return [a,cen]

class Kmeans:
    def __init__(self,k):
        self.k=k

    def fit(self,X):
        shape=X.shape
        cen=np.random.randint(0,shape[0],size=self.k)
        cen=X[cen]
        for j in range(100):
            least_cen=cen
            cen=np.tile(cen,[shape[0],1,1])
            cen=np.transpose(cen,[1,0,2])
            juli=cen-X
            juli=juli**2
            juli=juli.sum(axis=2)
            a=juli.argmin(axis=0)
            cen=[]
            for i in range(self.k):
                cen.append(X[a==i].mean(axis=0))
            cen=np.array(cen)
            self.cen=cen
            if (least_cen!=cen).sum()==0:
                break
        pass

    def predict(self,X):
        shape=X.shape
        cen=self.cen
        cen=np.tile(cen,[shape[0],1,1])
        cen=np.transpose(cen,[1,0,2])
        juli=cen-X
        juli=juli**2
        juli=juli.sum(axis=2)
        a=juli.argmin(axis=0)
        return a
        pass
    
kmean=Kmeans(5)

kmean.fit(X)

print(kmean.predict(X))