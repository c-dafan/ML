# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 17:36:41 2017

@author: 大帆
"""
import numpy as np
from ..base.Baseregress import Baseregree

class Knn_Regressor(Baseregree):
    def __init__(self,k):
        self.k=k
    def fit(self,X,y):
        self.train_x=X
        self.train_y=y
    def predict(self,X):
        if len(X.shape)==1:
            X=X.reshape([1,X.shape[0]])
        shape=self.train_x.shape
        test_x=np.tile(X,[shape[0],1,1])
        test_x=np.transpose(test_x,[1,0,2])
        juli=test_x-X
        juli=juli**2
        juli=juli.sum(axis=2)
        jusort=juli.argsort(axis=1)
        kne=jusort[:,:self.k]
#        print(kne)
        knc=self.train_y[kne]
        return knc.mean(axis=1)
        
