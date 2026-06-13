import numpy as np

class LinearRegression:
    def __init___(self,X,y):
        self.X=X
        self.y=y
    
    def loss(self,w):
        y_pred=self.X@w
        return np.mean(np.square(y_pred-self.y)) 
    
    def gradient(self,w,indices=None):
        if indices is None:
            X_batch=self.X
            y_batch=self.y
            n=len(self.X)
        else:
            X_batch=self.X[indices]
            y_batch=self.y[indices]
            n=len(indices)
        y_pred=X_batch@w
        err=y_pred-y_batch
        return (1/n)*np.transpose(X_batch)@err
