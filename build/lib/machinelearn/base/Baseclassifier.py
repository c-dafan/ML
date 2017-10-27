from ..score.score import accuracy_score

class BaseClassifer:
    def fit(self,X,y):
        pass
    def predict(self,X):
        pass
    def score(self,X,y):
        return accuracy_score(self.predict(X),y)