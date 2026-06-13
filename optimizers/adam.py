import numpy as np
from problems.logistics import *

class Adam:
    def __init__(self,X,y,w):
        self.X=X
        self.y=y
        self.L=LogisticRegression(self.X,self.y)
        
    def adam(self,w,indices,n,beta1=0.9,beta2=0.999,alpha=0.01,grad=None):
        if grad is None:
            g=self.L.gradient(w,indices)
        else:
            g=grad
        if not hasattr(self,'m') or self.m.shape!=g.shape:
            self.m=np.zeros_like(g)
            self.m2=np.zeros_like(g)

        self.m=self.m*beta1 + (1-beta1)*g
        self.m2=self.m2*beta2 + (1-beta2)*g*g
        
        # bias correction
        m_bias=self.m/(1-beta1**n)
        m2_bias=self.m2/(1-beta2**n)
        w-= alpha/(np.sqrt(m2_bias)+1e-10)*m_bias

        return w
    
    def step(self,w,indices,i):
        return self.adam(w,indices,i)