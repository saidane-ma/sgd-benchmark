import numpy as np
from problems.logistics import *

class Adam:
    def __init__(self,X,y,w):
        self.X=X
        self.y=y
        self.L=LogisticRegression(self.X,self.y)
        self.m=np.zeros_like(w)
        self.m2=np.zeros_like(w)
        
    def adam(self,w,indices,n,beta1=0.9,beta2=0.999,alpha=0.001):

        g=self.L.gradient(w,indices)
        self.m=self.m*beta1 + (1-beta1)*g
        self.m2=self.m2*beta2 + (1-beta2)*g*g
        
        # bias correction
        m_bias=self.m/(1-beta1**n)
        m2_bias=self.m2/(1-beta2**n)
        w-= alpha/(np.sqrt(m2_bias)+1e-10)*m_bias

        return w