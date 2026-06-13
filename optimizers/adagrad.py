import numpy as np
from problems.logistics import *
class Adagrad:
    def __init__(self,X,y,w):
        self.X=X
        self.y=y
        self.L=LogisticRegression(self.X,self.y)
        self.g=np.zeros_like(w)

    def adagrad(self,w,indices,alpha=0.1):
        self.g+=np.square(self.L.gradient(w,indices))
        w=w-(alpha/(np.sqrt(self.g + 1e-10)))*self.L.gradient(w,indices)
        
        return w

    def step(self,w,indices,i,alpha=0.1):
        return self.adagrad(w,indices,alpha)
    