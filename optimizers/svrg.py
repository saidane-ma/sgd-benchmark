import numpy as np
from problems import *

class SVRG:
    def __init__(self,problem):
        self.X=problem.X
        self.problem=problem
        
    def svrg(self,w,indices,alpha=0.001):
        w_tilde=w.copy()
        mu=self.problem.gradient(w_tilde)
        for i in range(2*len(self.X)):
            k= np.random.choice(len(self.X))
            g=self.problem.gradient(w,[k])-self.problem.gradient(w_tilde,[k])+mu
            w-= g*alpha
        return w 
    def step(self,w,indices,i,alpha=0.001,**kwargs):
        return self.svrg(w,indices,alpha)