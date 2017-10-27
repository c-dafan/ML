
import numpy as np
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score
from .knn.knn_Regressor import Knn_Regressor

data=load_boston()
X=data.data
y=data.target

knn=Knn_Regressor(k=2)
knn.fit(X,y)

yy=knn.predict(X)
print(r2_score(y,yy))
print(knn.score(X,y))

from .tree.DecisiontreeReg import DecisiontreeReg
tree=DecisiontreeReg()
tree.fit(X,y)
yy=tree.predict(X)
print(r2_score(y,yy))
print(tree.score(X,y))

from .regression.lr import Linear_regress
line=Linear_regress(lr=0.5,iter=300)
line.fit(X,y)
print(line.score(X,y))
print(r2_score(y,line.predict(X)))