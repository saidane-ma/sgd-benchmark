import numpy as np
from problems.logistics import *

np.random.seed(5)

class SAG:
    def __init__(self,X,y,w):
        self.X=X
        self.y=y
        self.L=LogisticRegression(X,y)
        self.g=np.zeros_like(self.X)

    def sag(self,w,indices,alpha=0.1):
        k=np.random.choice(indices)
        self.g[k-1]=self.L.gradient(w,[k])
        w-=alpha*np.mean(self.g,axis=0)
        return w
    def step(self,w,indices,i,alpha=0.1):
        return self.sag(w,indices,alpha)