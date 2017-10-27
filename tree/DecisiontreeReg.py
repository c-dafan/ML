# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 22:26:52 2017

@author: 大帆
"""

import numpy as np
from ..base.Baseregress import Baseregree

class DecisiontreeReg(Baseregree):
    def __init__(self,min_split=10):
        self.min_split=min_split

    def Binsplitdata(self,dataset,index,value):
        smaller=dataset[dataset[:,index]<=value]
        bigger=dataset[dataset[:,index]>value]
        return smaller,bigger
        
    def Regloss(self,dataset):
        a=np.mean(dataset[:,-1])
        return np.square(dataset[:,-1]-a).sum()
        
    def ChooseBestsplit(self,dataset):
        if np.unique(dataset[:,-1]).shape[0] == 1:
            return None, dataset[:,-1].mean()
        if dataset.shape[0]<10:
            return None, dataset[:,-1].mean()
        loss=np.inf
        bestindex=0
        bestsplit=0
        for i in range(dataset.shape[1]-1):
            undata=np.unique(dataset[:,i])
            for val in undata:
                sm,bg=self.Binsplitdata(dataset,i,val)
                if sm.shape[0]<10 or bg.shape[0]<10:
                    continue
                loss1=self.Regloss(sm)+self.Regloss(bg)
                if loss>loss1:
                    loss=loss1
                    bestindex=i
                    bestsplit=val
        if abs(loss-self.Regloss(dataset))<1:
            return None,dataset[:,-1].mean()
        sm,bg=self.Binsplitdata(dataset,bestindex,bestsplit)
        if sm.shape[0]<10 or bg.shape[0]<10:
            return None,dataset[:,-1].mean()
        return (bestindex,bestsplit)
    
    def CreateTree(self,dataset):
        bestindex,bestsplit=self.ChooseBestsplit(dataset)
        if bestindex==None:
            return bestsplit
        sm,bg=self.Binsplitdata(dataset,bestindex,bestsplit)
        Tree={}
        Tree['Child']={'left':self.CreateTree(sm),'right':self.CreateTree(bg)}
        Tree['Index']=bestindex
        Tree['Bestsplit']=bestsplit
        return Tree
    def fit(self,x,y):
        dataset=np.c_[x,y]
        self.tree=self.CreateTree(dataset)
        
    def _predict(self,dataset,Tree):
        if type(Tree)!=dict:
            return(Tree)
        index=Tree['Index']
        bestsplit=Tree['Bestsplit']
        if dataset[index]<=bestsplit:
            return self._predict(dataset,Tree['Child']['left'])
        else:
            return self._predict(dataset,Tree['Child']['right'])
    
    def predict(self,dataset):
        return np.apply_along_axis(lambda x:self._predict(x,self.tree),1,dataset)
#        res=[]
#        for i in dataset:
#            res.append(self._predict(i,self.tree))
#        return np.array(res)


