# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 16:00:26 2017

@author: 大帆
"""

import numpy as np
from ..base.Baseclassifier import BaseClassifer



class Bayer(BaseClassifer):
    def __init__(self):
        pass
    def Prior_probability(self,dataset,index,x_val,y_val):
        x=dataset[dataset[:,-1]==y_val]
        x=x[x[:,index]==x_val]
        y=dataset[dataset[:,-1]==y_val]
        return x.shape[0]/y.shape[0]
    def fit(self,X,y):
        dataset=np.c_[X,y]
        probability,loc,label =self._fit(dataset)
        self.probability=probability
        self.loc=loc
        self.label=label
    def _fit(self,dataset):
        probability=[]
        loc={}
        tofirst=0
        for i in range(dataset.shape[1]-1):
            b=np.unique(dataset[:,i])
            if b.shape[0]<=10:
                loc[i]=[{},tofirst]
                tofirst+=b.shape[0]
                for l_v,val in enumerate(b):
                    loc[i][0][val]=l_v
                    Prior_pro=[]
                    for y_val in np.unique(dataset[:,-1]):
                        Prior_pro.append(self.Prior_probability(dataset,i,val,y_val))
                    probability.append(Prior_pro)
            else:
                y=[]
                loc[i]=[' ',tofirst]
                tofirst+=1
                for y_val in np.unique(dataset[:,-1]):
                    y.append([dataset[dataset[:,-1]==y_val][:,i].mean(),\
                                        dataset[dataset[:,-1]==y_val][:,i].std()])
                probability.append(y)
        y=[]
        for y_val in np.unique(dataset[:,-1]):
            y.append(dataset[dataset[:,-1]==y_val].shape[0]/dataset.shape[0])
        probability.append(y)
        label={}
        for i,lab in enumerate(np.unique(dataset[:,-1])):
            label[i]=lab
        return probability,loc,label
    def _predict(self,X,probability,loc,label):
        res=[]
        for i in range(3):
            a=probability[-1][i]
            for r in range(X.shape[0]):
#                if r==0:
#                    a*=probability[loc[r][0][X[r]]][i]
#                else:
                if type(loc[i][0])==dict:
                    a*=probability[loc[r][1]+loc[r][0][X[r]]][i]
                else:
#                    if r==0:
#                        m,s=probability[0][i]
#                        print(m,s)
#                        a*=1/(2.50662827*s)*np.exp(-(X[r]-m)**2/(2*s**2))
#                    else:
                    m,s=probability[loc[r][1]][i]
#                    print(m,s)
                    a*=1/(2.50662827*s)*np.exp(-(X[r]-m)**2/(2*s**2))
            res.append(a)
        return label[np.argmax(res)]

    def predict(self,X):
        return np.apply_along_axis(lambda x:self._predict(x,self.probability,\
                                                          self.loc,self.label), 1,X)










