import numpy as np
from problems.logistics import *

class SVRG:
    def __init__(self,X,y,w):
        self.X=X
        self.y=y
        self.L=LogisticRegression(X,y)
        
    def svrg(self,w,indices,alpha=0.01):
        w_tilde=w
        mu=self.L.gradient(w_tilde)
        for i in range(2*len(self.X)):
            k= np.random.choice(indices)
            g=self.L.gradient(w,[k])-self.L.gradient(w_tilde,[k])+mu
            w-= g*alpha
        return w