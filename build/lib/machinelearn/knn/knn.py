import numpy as np
from ..base.Baseclassifier import BaseClassifer

class Knn_Classifier(BaseClassifer):
    def __init__(self,k=3):
        self.k=k
    def fit(self,X,y):
        self.train_x=X
        self.train_y=y
    def predict(self,X):
        if len(X.shape)==1:
            X=X.reshape([1,4])
        shape=self.train_x.shape
        shape_t=X.shape
        test_x=np.tile(X,[shape[0],1,1])
        test_x=np.transpose(test_x,[1,0,2])
        juli=test_x-X
        juli=juli**2
        juli=juli.sum(axis=2)
        jusort=juli.argsort(axis=1)
        kne=jusort[:,:self.k]
        knc=self.train_y[kne]
        result=[]
        for row in knc:
            result.append(row[row[1].argmax()])
        return np.array(result).reshape((shape_t[0],1))
