import numpy as np

from problems.logistics import *

class SGD:
    def __init__(self,X,y):
        self.X=X
        self.y=y
    
    def polyak(self,w,indices,beta=0.9):
        L=LogisticRegression(self.X,self.y)
        v=np.zeros_like(w)
        for i in range(len(self.X)):
            v=beta*v+(1-beta)*L.gradient(w,indices)
            w-=v
        return w
    
    def momentum(self,w,indices,beta=0.9):
        L=LogisticRegression(self.X,self.y)
        v=np.zeros_like(w)
        for i in range(len(self.X)):
            v=beta*v+L.gradient(w,indices)
            w-=(1-beta)*v
        return w

    def nestrov(self,w,indices,beta=0.9):
        L=LogisticRegression(self.X,self.y)
        v=np.zeros_like(w)
        for i in range(len(self.X)):
            v=beta*v+(1-beta)*L.gradient(w-v,indices)
            w-=v
        return v
        