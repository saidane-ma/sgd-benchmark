import numpy as np

from problems import *

class SGD:
    def __init__(self,problem,mode,w):
        self.v_polyak=np.zeros_like(w)
        self.v_nestrov=np.zeros_like(w)
        self.v_momentum=np.zeros_like(w)
        self.problem=problem
        self.mode=mode
    
    def polyak(self,w,indices,alpha=0.1,beta=0.9):
        self.v_polyak=beta*self.v_polyak+alpha*self.problem.gradient(w,indices)
        w-=self.v_polyak
        return w
    
    def momentum(self,w,indices,alpha=0.1,beta=0.9):
        self.v_momentum=beta*self.v_momentum+self.problem.gradient(w,indices)
        w-=alpha*self.v_momentum
        return w

    def nestrov(self,w,indices,alpha=0.1,beta=0.99):
        self.v_nestrov=beta*self.v_nestrov+alpha*self.problem.gradient(w-beta*self.v_nestrov,indices)
        w-=self.v_nestrov
        return w
    
    def step(self,w,indices,i,alpha=0.001,beta1=0.9,beta2=0.99,**kwargs):
        if self.mode=="polyak": return self.polyak(w,indices,alpha,beta1)
        elif self.mode=="nesterov" : return self.nestrov(w,indices,0.001,beta2)
        else : return w - alpha*self.problem.gradient(w,indices)