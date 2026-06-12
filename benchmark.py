import numpy as np
import matplotlib.pyplot as plt
from data import*
from problems.logistics import*
from optimizers.sgd import *

#parameters
n_samples=1000
n_features=2
alpha=0.01
n_iterations=500
batch_size=10
beta=0.9

#data generation
X, y=generate_gaussian_blobs(n_features,n_samples)
np.random.seed(5)

L=LogisticRegression(X,y)
SGD=SGD(X,y,np.random.randn(n_features+1))

w0=np.random.randn(n_features+1)
w1=np.random.randn(n_features+1)
w2=np.random.randn(n_features+1)
w3=np.random.randn(n_features+1)

#storing for plot
history=[]
history_SGDM=[]
history_Adam=[]
history_Adagrad=[]
history_Nesterov=[]
history_polyak=[]


#benchmarking loop
for i in range(n_iterations):
    indices=np.random.choice(n_samples,size=(batch_size,),replace=False)
    grad=L.gradient(w0,indices)
    w0=w0-alpha*grad
    history.append(L.loss(w0))

    w1=SGD.momentum(w1,indices,alpha,beta)
    history_SGDM.append(L.loss(w1))

    w2=SGD.polyak(w2,indices,alpha,beta)
    history_polyak.append(L.loss(w2))
    
    w3=SGD.nestrov(w3,indices,alpha,beta)
    history_Nesterov.append(L.loss(w3))
    

#plotting

fig, axs = plt.subplots(2, 2)

axs[0, 0].plot(history) 
axs[0, 0].set_title("SGD")
axs[0, 0].set_ylim(0.5,1.5)

axs[0, 1].plot(history_Nesterov)
axs[0, 1].set_title("Nesterov")
axs[0, 1].set_ylim(0.5,1.5)

axs[1, 0].plot(history_polyak)
axs[1, 0].set_title("Polyak")
axs[1, 0].set_ylim(0.5,1.5)

axs[1, 1].plot(history_SGDM)
axs[1, 1].set_title("Standard Momentum")
axs[1, 1].set_ylim(0.5,1.5)


fig.suptitle('SGD Variants')


plt.tight_layout()
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.show()
