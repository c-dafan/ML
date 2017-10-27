# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 19:48:56 2017

@author: 大帆
"""
from ..base.Baseclassifier import BaseClassifer
import numpy as np

class Cart_Decision(BaseClassifer):
    def __init__(self,):
        pass

    def gini(self,dataset,feature=-1):
        label_values =np.unique(dataset[:,feature],return_counts=True)
        gini_sub=0.0
        for val,counts in zip(*label_values):
            gini_sub+=(counts/dataset.shape[0])**2
        return 1-gini_sub
        
    def Binsplitdata(self,dataset,index,value):
        smaller=dataset[dataset[:,index]<=value]
        bigger=dataset[dataset[:,index]>value]
        return smaller,bigger
    def Gini_gain(self,dataset,feature,uncon=True,val=None):
        if uncon:
            equit=dataset[dataset[:,feature]==val]
            notequit=dataset[dataset[:,feature]!=val]
            gini_sub=equit.shape[0]/dataset.shape[0]*self.gini(equit)+\
                    notequit.shape[0]/dataset.shape[0]*self.gini(notequit)
            return gini_sub
        else:
            sm,bg=self.Binsplitdata(dataset,feature,val)
            gini_sub=sm.shape[0]/dataset.shape[0]*self.gini(sm)+\
                    bg.shape[0]/dataset.shape[0]*self.gini(bg)
            return gini_sub
            
    def Select_Best(self,dataset,feature1,feature2):
        loss=np.inf
        best_index=0
        best_split=0
        uncon=True
        for feature in feature1:
            undata=np.unique(dataset[:,feature])
            if undata.shape[0]==2:
                loss1=self.Gini_gain(dataset,feature,uncon=True,val=undata[0])
    #            print(loss1)
                best_index=feature
                best_split=undata[0]
            else:
                for val in undata:
                    loss1=self.Gini_gain(dataset,feature,uncon=True,val=val)
    #                print(loss1)
                    if loss1<loss:
                        loss=loss1
                        best_index=feature
                        best_split=val
                        uncon=True
        for feature in feature2:
            undata=np.unique(dataset[:,feature])
            for val in undata[:-1]:
                loss1=self.Gini_gain(dataset,feature,uncon=False,val=val)
    #            print(loss1)
                if loss1<loss:
                    loss=loss1
                    best_index=feature
                    best_split=val
                    uncon=False
        return best_index,best_split,uncon
    
    def majority_label(self,labels):
        return max(zip(*np.unique(labels,return_counts=True)),key=lambda x:x[1])[0]
    def CreateTree(self,dataset,feature1,feature2):
        if np.unique(dataset[:,-1]).shape[0]==1:
            return dataset[0,-1]
        if len(feature1)==0 and len(feature2)==0:
            return self.majority_label(dataset[:,-1])
        best_index,best_split,uncon=self.Select_Best(dataset,feature1,feature2)
        Tree={}
        if uncon:
            Tree['best_index']=best_index
            Tree['best_split']=best_split
            Tree['uncon']=uncon
            equit=dataset[dataset[:,best_index]==best_split]
            notequit=dataset[dataset[:,best_index]!=best_split]
            Tree['Child']={}
            Tree['Child']['left']=self.CreateTree(equit,\
                    [x for x in feature1 if x != best_index],feature2)
            if np.unique(notequit[:,best_index]).shape[0]!=1:
                Tree['Child']['right']=self.CreateTree(notequit,feature1,feature2)
            else:
                Tree['Child']['right']=self.CreateTree(notequit,\
                    [x for x in feature1 if x != best_index],feature2)
        else:
            Tree['best_index']=best_index
            Tree['best_split']=best_split
            Tree['uncon']=uncon
            sm,bg=self.Binsplitdata(dataset,best_index,best_split)
            Tree['Child']={}
            Tree['Child']['left']=self.CreateTree(sm,feature1,feature2)
            Tree['Child']['right']=self.CreateTree(bg,feature1,feature2)
        return Tree
    def fit(self,X,y):
        dataset=np.c_[X,y]
        features1=[]
        features2=[]
        for i in range(dataset.shape[1]-1):
            if np.unique(dataset[:,i]).shape[0]<=10:
                features1.append(i)
            else:
                features2.append(i)
        self.tree=self.CreateTree(dataset,features1,features2)
    def _predict(self,dataset,Tree):
        if type(Tree)!=dict:
            return(Tree)
        index=Tree['best_index']
        bestsplit=Tree['best_split']
        uncon=Tree['uncon']
        if uncon:
            if dataset[index]==bestsplit:
                return self._predict(dataset,Tree['Child']['left'])
            else:
                return self._predict(dataset,Tree['Child']['right'])
        else:
            if dataset[index]<=bestsplit:
                return self._predict(dataset,Tree['Child']['left'])
            else:
                return self._predict(dataset,Tree['Child']['right'])
    def predict(self,X):
        return np.apply_along_axis(lambda x:self._predict(x,self.tree),1,X)
        
# dataset=[[1,1,'yes'],
#          [1,1,'yes'],
#          [1,0,'no'],
#          [1,0,'no'],
#          [0,1,'no'],
#          [0,1,'no'],
#          [0,0,'yes'],
#          [0,0,'yes'],
#          [0,0,'yes']
#          ]
# dataset=np.array(dataset)
# X=dataset[:,:-1]
# y=dataset[:,-1]

#from sklearn.datasets import load_iris
#data=load_iris()
#X=data.data
#y=data.target
#dataset=np.c_[X,y]
# Det=Cart_Decision()
# Det.fit(X,y)
# print(Det.predict(X))