# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 20:46:02 2017

@author: 大帆
"""

import numpy as np
from ..base.Baseclassifier import BaseClassifer
from .Cart_ import Cart_Decision

class DecisiontreeClassifier(BaseClassifer):
    def __init__(self,criterion="gini"):
        if criterion=='gini':
            self.model=Cart_Decision()
        else:
            self.model=EntropyClassifier(criterion=criterion)
    def fit(self,X,y):
        self.model.fit(X,y)
    def predict(self,X):
        return self.model.predict(X)
    def score(self,X,y):
        return self.model.score(X,y)

class EntropyClassifier(BaseClassifer):
    def __init__(self,criterion='c45'):
        self.criterion=criterion
        pass
    def entropy(self,dataset,fea=-1):
        labels = dataset[:,fea]
        h = 0.0
        for k,v in zip(*np.unique(labels,return_counts=True)):
            h -= v / labels.shape[0] * np.log2([v / labels.shape[0]])[0]
        return h
    def Binsplitdata(self,dataset,index,value):
        smaller=dataset[dataset[:,index]<=value]
        bigger=dataset[dataset[:,index]>value]
        return smaller,bigger
    def Regloss(self,dataset):
        a=np.mean(dataset[:,-1])
        return np.square(dataset[:,-1]-a).sum()
    def info_gain(self,dataset, feature,val=None):
        if val==None:
            feature_values =np.unique( dataset[:,feature])
            entropy_sub_dataset = 0.0
            for val in feature_values:
                sub_dataset=dataset[dataset[:,feature]==val]
                entropy_sub_dataset += sub_dataset.shape[0]/dataset.shape[0]\
                            * self.entropy(sub_dataset)
            return self.entropy(dataset) - entropy_sub_dataset
        else:
            sm,bg=self.Binsplitdata(dataset,feature,val)
            hda=-sm.shape[0]/dataset.shape[0]*self.entropy(sm)-\
                       bg.shape[0]/dataset.shape[0]*self.entropy(bg)
            hd=self.entropy(dataset)
            return (hd-hda)
    def info_gain_ratio(self,dataset, feature,val=None):
        if val==None:
            feature_values =np.unique( dataset[:,feature])
            entropy_sub_dataset = 0.0
            for val in feature_values:
                sub_dataset=dataset[dataset[:,feature]==val]
                entropy_sub_dataset += sub_dataset.shape[0]/dataset.shape[0]\
                            * self.entropy(sub_dataset)
            Had=self.entropy(dataset,feature)
            return (self.entropy(dataset) - entropy_sub_dataset)/Had
        else:
            sm,bg=self.Binsplitdata(dataset,feature,val)
            hda=-sm.shape[0]/dataset.shape[0]*self.entropy(sm)-\
                       bg.shape[0]/dataset.shape[0]*self.entropy(bg)
            hd=self.entropy(dataset)
            had=-np.log2(sm.shape[0]/dataset.shape[0])\
                        -np.log2(bg.shape[0]/dataset.shape[0])
            return (hd-hda)/had

    def select_feature(self,dataset, features1,features2):
        if dataset.shape[0]<=1*2:
            return None,self.majority_label(dataset[:,-1])
        if self.criterion=='id3':
            criterion_select=self.info_gain
        elif self.criterion=='c45':
            criterion_select=self.info_gain_ratio
        if len(features1)!=0:
            info_gains = [(criterion_select(dataset, x), x) for x in features1]
            loss=max(info_gains,key=lambda x:x[0])[0]
            bestindex=max(info_gains,key=lambda x:x[0])[1]
        else:
            loss=-np.inf
            bestindex=0
        bestsplit=0
        uncon=True
        c=0
        for i in features2:
            undata=np.unique(dataset[:,i])
            if undata.shape[0]==1:
                c+=1
            for val in undata:
                sm,bg=self.Binsplitdata(dataset,i,val)
                if sm.shape[0]<1 or bg.shape[0]<1:
                    continue
                loss1=criterion_select(dataset,i,val)
#                print(loss1)
                if loss<loss1:
                    loss=loss1
                    bestindex=i
                    bestsplit=val
                    uncon=False
#        print(loss,bestindex,bestsplit)
        if c==dataset.shape[1]-1:
            return None,self.majority_label(dataset[:,-1])
        if uncon:
            return max(info_gains)[1],None
        return bestindex,bestsplit
     
    def majority_label(self,labels):
        return max(zip(*np.unique(labels,return_counts=True)),key=lambda x:x[1])[0]

    def build_tree(self,dataset, features1,features2):
        labels = dataset[:,-1]
        features=features1+features2
        if np.unique(labels).shape[0]==1:
            return {'label': labels[0]}
        if len(features) == 0:
            return {'label': self.majority_label(labels)}
        best_feature,best_split = self.select_feature(dataset, features1,features2)
        if best_split==None:
            tree = {'feature': best_feature, 'children': {}}
            best_feature_values = np.unique(dataset[:,best_feature])
            for val in best_feature_values:
                sub_dataset =dataset[ dataset[:,best_feature]==val]
                if len(sub_dataset) == 0:
                    tree['children'][val] = {
                        'label': self.majority_label(labels)}
                else:
                    tree['children'][val] = self.build_tree(
                        sub_dataset, [x for x in features1 if x != best_feature],features2)
        else:
            if best_feature==None:
                return {'label': best_split}
            tree = {'feature': best_feature, 'children': {},'best_split':best_split}
            sm,bg=self.Binsplitdata(dataset,best_feature,best_split)
            tree['children']['left']=self.build_tree(sm,features1,[x for x in features2 if x != best_feature])
            tree['children']['right']=self.build_tree(bg,features1,[x for x in features2 if x != best_feature])
        return tree
        
    def fit(self,X,y):
        dataset=np.c_[X,y]
        feature1=[]
        feature2=[]
        for i in range(dataset.shape[1]-1):
            if np.unique(dataset[:,i]).shape[0]<=10:
                feature1.append(i)
            else:
                feature2.append(i)
        self.tree=self.build_tree(dataset, feature1,feature2)
    def _predict(self,tree,x):
        if 'feature' in tree:
            if 'best_split' in tree:
                if x[tree['feature']]<=tree['best_split']:
                    return self._predict(tree['children']['left'], x)
                else:
                    return self._predict(tree['children']['right'], x)
            else:
                return self._predict(tree['children'][x[tree['feature']]], x)
        else:
            return tree['label']
    def predict(self,X):
        return np.apply_along_axis(lambda x:self._predict(self.tree,x),1,X)

#dataset = [x.strip().split('  ') for x in open('lenses.data')]
#dataset=np.array(dataset)
#dataset=dataset.astype('int')
# dataset=[[1,1,'yes'],
#          [1,1,'yes'],
#          [1,0,'no'],
#          [0,1,'no'],
#          [0,1,'no'],
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
# DecisionTree=DecisiontreeClassifier(criterion='c45')
# DecisionTree.fit(X,y)
# print(DecisionTree.predict(X))
