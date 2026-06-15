import numpy as np
from problems.logistics import *

class Adagrad:
    def __init__(self,w,problem):
        self.problem=problem
        self.g=np.zeros_like(w)

    def adagrad(self,w,indices,alpha=0.1):
        self.g+=np.square(self.problem.gradient(w,indices))
        w=w-(alpha/(np.sqrt(self.g + 1e-10)))*self.problem.gradient(w,indices)
        
        return w

    def step(self,w,indices,i,alpha=0.1,**kwargs):
        return self.adagrad(w,indices,alpha)
    