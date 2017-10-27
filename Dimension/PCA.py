import numpy as np

from sklearn.datasets import load_boston
import matplotlib.pyplot as plt

data=load_boston()
X=data.data
y=data.target

def pca(X,n_d):
    xme=X.mean(axis=0)
    x_me=X-xme
    xx=np.dot(x_me.T,x_me)
    a,b=np.linalg.eig(xx)
    return np.dot(x_me,b[:,:n_d]),a[:n_d].sum()/a.sum()
    
class PCA:
    def __init__(self,n_d):
        self.n_d=n_d
        pass
    def fit(self,X):
        xme=X.mean(axis=0)
        x_me=X-xme
        xx=np.dot(x_me.T,x_me)
        self.a,self.b=np.linalg.eig(xx)
    def predict(self,X):
        xme=X.mean(axis=0)
        x_me=X-xme
        return np.dot(x_me,self.b[:,:self.n_d])
    def fit_predict(self,X):
        xme=X.mean(axis=0)
        x_me=X-xme
        xx=np.dot(x_me.T,x_me)
        self.a,self.b=np.linalg.eig(xx)
        return np.dot(x_me,self.b[:,:self.n_d])
    def plot(self):
        a=self.a
        t=1-a.cumsum()/a.sum()
        plt.figure()
        plt.plot(np.arange(t.shape[0]),t)
        plt.show()
