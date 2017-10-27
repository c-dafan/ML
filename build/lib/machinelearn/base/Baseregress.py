from ..score.score import r2_score

class Baseregree:
    def fit(self,X,y):
        pass
    def predict(self,X):
        pass
    def score(self,X,y):
        return r2_score(self.predict(X),y)