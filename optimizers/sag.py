import numpy as np
from problems import *

np.random.seed(5)

class SAG:
    def __init__(self,problem,w_init):
        self.X=problem.X
        self.problem=problem
        self.g=np.zeros((len(self.X),len(w_init)))
        for i in range(len(self.X)):
            self.g[i]=self.problem.gradient(w_init, [i])

    def sag(self,w,indices=None,alpha=0.008):
        k=np.random.choice(len(self.X))
        self.g[k]=self.problem.gradient(w,[k])
        w-=alpha*np.mean(self.g,axis=0)
        return w
    def step(self,w,indices,i,alpha=0.008,**kwargs):
        return self.sag(w,indices,alpha)