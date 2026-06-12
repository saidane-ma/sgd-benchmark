import numpy as np

class LogisticRegression:
    def __init__(self,X,y):
        self.X=X
        self.y=y.T
    
    def sigmoid(self,z):
        return 1/(1+np.exp(-z))
    
    def loss(self,w):
        z=self.X@w
        y_pred=self.sigmoid(z)
        n=len(self.X)
        s=-self.y*np.log(y_pred+(1e-10))-(1-self.y)*np.log(1-y_pred+(1e-10))

        return np.sum(s)/n

    def gradient(self,w,indices=None):
        if indices is None:
            X_batch=self.X
            y_batch=self.y
            n=len(self.X)
        else:
            X_batch=self.X[indices]
            y_batch=self.y[indices]
            n=len(indices)
        y_pred=self.sigmoid(X_batch@w)
        err=y_pred-y_batch
        return (1/n)*np.transpose(X_batch)@err
