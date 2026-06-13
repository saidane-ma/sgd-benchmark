import numpy as np
from logistics import *

class NN:
    def __init__(self,X,y):
        self.X=X
        self.y=y
        self.L=LogisticRegression(X,y)

    def sigmoid_NN(self,w,w_):
        z=self.X@w
        y1=self.L.sigmoid(z)
        b=np.array([1]*len(y1))
        y1=y1.insert(y1,0,b,axis=1)
        return y1@w_

    def ReLU_NN(self,w,w_):
        z=self.X@w
        y1=np.maximum(0,z)
        b=np.array([1]*len(y1))
        y1=y1.insert(y1,0,b,axis=1)
        return y1@w_
    
    def tanh_NN(self,w,w_):
        z=self.X@w
        y1=np.tanh(z)
        b=np.array([1]*len(y1))
        y1=y1.insert(y1,0,b,axis=1)
        return y1@w_
    