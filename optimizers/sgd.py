import numpy as np

from problems.logistics import *

class SGD:
    def __init__(self,X,y,w,mode):
        self.X=X
        self.y=y
        self.v_polyak=np.zeros_like(w)
        self.v_nestrov=np.zeros_like(w)
        self.v_momentum=np.zeros_like(w)
        self.mode=mode
    
    def polyak(self,w,indices,alpha=0.1,beta=0.9):
        L=LogisticRegression(self.X,self.y)
        self.v_polyak=beta*self.v_polyak+alpha*L.gradient(w,indices)
        w-=self.v_polyak
        return w
    
    def momentum(self,w,indices,alpha=0.1,beta=0.9):
        L=LogisticRegression(self.X,self.y)
        self.v_momentum=beta*self.v_momentum+L.gradient(w,indices)
        w-=alpha*self.v_momentum
        return w

    def nestrov(self,w,indices,alpha=0.1,beta=0.99):
        L=LogisticRegression(self.X,self.y)
        self.v_nestrov=beta*self.v_nestrov+alpha*L.gradient(w-beta*self.v_nestrov,indices)
        w-=self.v_nestrov
        return w
    
    def step(self,w,indices,i,alpha=0.1,beta1=0.9,beta2=0.99,mode="polyak"):
        if mode=="polyak": return self.polyak(w,indices,alpha,beta1)
        elif mode=="nesterov" : return self.nestrov(w,indices,0.01,beta2)
        else : return w-alpha*self.L.gradient(w,indices)