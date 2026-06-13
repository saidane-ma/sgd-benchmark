import numpy as np
from problems.logistics import *

class SVRG:
    def __init__(self,X,y,w):
        self.X=X
        self.y=y
        self.L=LogisticRegression(X,y)
        
    def svrg(self,w,indices,alpha=0.001):
        w_tilde=w.copy()
        mu=self.L.gradient(w_tilde)
        for i in range(2*len(self.X)):
            k= np.random.choice(len(self.X))
            g=self.L.gradient(w,[k])-self.L.gradient(w_tilde,[k])+mu
            w-= g*alpha
        return w 
    def step(self,w,indices,i,alpha=0.001):
        return self.svrg(w,indices,alpha)