import numpy as np
from problems.logistics import *

np.random.seed(5)

class SAG:
    def __init__(self,X,y,w):
        self.X=X
        self.y=y
        self.L=LogisticRegression(X,y)
        self.g=np.zeros((len(self.X),len(w)))
        for i in range(len(self.X)):
            self.g[i]=self.L.gradient(w, [i])

    def sag(self,w,indices,alpha=0.003):
        k=np.random.choice(len(self.X))
        self.g[k]=self.L.gradient(w,[k])
        w-=alpha*np.mean(self.g,axis=0)
        return w
    def step(self,w,indices,i,alpha=0.003):
        return self.sag(w,indices,alpha)