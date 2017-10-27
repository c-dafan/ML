# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 19:38:26 2017

@author: 大帆
"""

import numpy as np

class SVD:
    def __init__(self,n=3):
        self._n=3
    def fit_tr(self,X):
        u,s,t=np.linalg.svd(X)
        u=u[:,:self._n]
        s=np.eye(self._n)*s[:self._n]
        t=t[:self._n,:]
        return u,s,t