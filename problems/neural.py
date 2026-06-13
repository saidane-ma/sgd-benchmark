import numpy as np
from problems.logistics import *

class NN:
    def __init__(self,X,y):
        self.X=X
        self.y=y
        self.L=LogisticRegression(X,y)

    def flatten(self,w,w_):
        return np.concatenate([w.flatten(),w_.flatten()])
    
    def unflatten(self,w,shape,shape_):
        return w[:(shape[0]*shape[1])].reshape(shape), w[(shape[0]*shape[1]):].reshape(shape_)
    
    def sigmoid_NN(self,w,w_):
        z=self.X@w
        y1=self.L.sigmoid(z)
        b=np.array([1]*len(y1))
        y1_=np.insert(y1,0,b,axis=1) #bias term insertion
        y_pred=self.L.sigmoid(y1_@w_) #necessary for classification problems
        err=(y_pred-self.y.reshape(-1,1))/len(self.y) #normalization 

        grad_=np.transpose(y1_)@err
        err =(err@np.transpose(w_[1:]))*self.L.sigmoid(z)*(1-self.L.sigmoid(z)) #Hadamard multiplication (not matrix mult)
        grad=np.transpose(self.X)@err
        return grad, grad_,y_pred
    
    def ReLU_NN(self,w,w_):
        z=self.X@w
        y1=np.maximum(0,z)
        b=np.array([1]*len(y1))
        y1_=np.insert(y1,0,b,axis=1)

        y_pred=self.L.sigmoid(y1@w_) #necessary for classification problems
        err=(y_pred-self.y.reshape(-1,1))/len(self.y) #normalization 

        grad_=np.transpose(y1_)@err
        err =(err@np.transpose(w_[1:]))*(z>0) #ReLU derivative : 0 if z<=0, 1 else
        grad=np.transpose(self.X)@err

        return grad, grad_,y_pred
    
    def tanh_NN(self,w,w_):
        z=self.X@w
        y1=np.tanh(z)
        b=np.array([1]*len(y1))
        y1_=np.insert(y1,0,b,axis=1)
        y_pred=self.L.sigmoid(y1@w_) #necessary for classification problems

        err=(y_pred-self.y.reshape(-1,1))/len(self.y) #normalization 

        grad_=np.transpose(y1_)@err
        err =(err@np.transpose(w_[1:]))*(1-y1**2)
        grad=np.transpose(self.X)@err
        return grad, grad_,y_pred
    
    def loss(self,y_pred):
        n=len(self.X)
        s=-self.y*np.log(y_pred+(1e-10))-(1-self.y)*np.log(1-y_pred+(1e-10))
        return np.sum(s)/n