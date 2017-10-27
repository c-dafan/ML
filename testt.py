# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 16:55:16 2017

@author: 大帆
"""
import numpy as np
from .bayers.Bayer import Bayer

from .knn.knn import Knn_Classifier
from sklearn.datasets import load_iris
from .tree.DecisionTreeClassifier import DecisiontreeClassifier

data=load_iris()
X=data.data
y=data.target
dataset=np.c_[X,y]

bayer=Bayer()
bayer.fit(X,y)
# print(bayer.predict(dataset[:,:-1]))
print(bayer.score(X,y))

knn_=Knn_Classifier(5)
knn_.fit(X,y)
print(knn_.score(X,y))

from .regression.Logist_reg import Logistic_Reg
Log=Logistic_Reg()
x=X[:100]
Log.fit(x,y[:100])
print(Log.score(x,y[:100]))


Det=DecisiontreeClassifier(criterion="gini")
Det.fit(X,y)
print(Det.score(X,y))

DecisionTree=DecisiontreeClassifier(criterion='c45')
DecisionTree.fit(X,y)
print(DecisionTree.score(X,y))
