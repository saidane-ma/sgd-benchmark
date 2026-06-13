import numpy as np
import matplotlib.pyplot as plt
from data import*
from problems.logistics import*
from optimizers.sgd import *
from optimizers.adagrad import*
from optimizers.adam import *
from optimizers.sag import *

#parameters
n_samples=100
n_features=2
alpha=0.01
n_iterations=5000
batch_size=10
beta=0.9

#data generation
X, y=generate_gaussian_blobs(n_features,n_samples)
np.random.seed(5)

L=LogisticRegression(X,y)
SGD=SGD(X,y,np.random.randn(n_features+1))
Adagrad=Adagrad(X,y,np.random.randn(n_features+1))
Adam=Adam(X,y,np.random.randn(n_features+1))
SAG=SAG(X,y,np.random.randn(n_features+1))

w= np.random.randn(n_features + 1)

# 2. Distribute identical copies to every algorithm
w0 = np.copy(w) # SGD
w1 = np.copy(w) # Adagrad
w2 = np.copy(w) # Polyak
w3 = np.copy(w) # Nesterov
w4 = np.copy(w) # Adam
w5 = np.copy(w) # SAG


#storing for plot
history=[]
history_SGDM=[]
history_Adam=[]
history_Adagrad=[]
history_Nesterov=[]
history_polyak=[]
history_sag=[]

#benchmarking loop
for i in range(n_iterations):
    indices=np.random.choice(n_samples,size=(batch_size,),replace=False)

    grad=L.gradient(w0,indices)
    w0=w0-alpha*grad
    history.append(L.loss(w0))

    w1=Adagrad.adagrad(w1,indices,alpha)
    history_Adagrad.append(L.loss(w1))

    w2=SGD.polyak(w2,indices,alpha,beta)
    history_polyak.append(L.loss(w2))
    
    w3=SGD.nestrov(w3,indices,alpha,beta)
    history_Nesterov.append(L.loss(w3))
    
    w4=Adam.adam(w4,indices,i+1)
    history_Adam.append(L.loss(w4))

    w5=SAG.sag(w5,indices)
    history_sag.append(L.loss(w5))


print(history_sag[0])
print(history_polyak[0])

#plotting
print("Plotting")

fig, axs = plt.subplots(3, 3)

axs[0, 0].plot(history) 
axs[0, 0].set_title("SGD")

axs[0, 1].plot(history_Nesterov)
axs[0, 1].set_title("Nesterov")

axs[1, 0].plot(history_polyak)
axs[1, 0].set_title("Polyak")

axs[1, 1].plot(history_Adagrad)
axs[1, 1].set_title("Adagrad")

axs[0,2].plot(history_Adam)
axs[0,2].set_title("Adam")

axs[2,0].plot(history_sag)
axs[2,0].set_title("SAG")   
#axs[2,0].set_ylim(0.6,1)

#axs[2,2].plot(history_svrg)
axs[2,2].set_title("SVRG")

"""axs[1,2].plot(Adam.history)
axs[1,2].set_title("Adam norm")
axs[1,2].set_xlabel("Iterations")
axs[1,2].set_ylabel("Norm of gradient")
axs[1,2].set_ylim(0,1)"""

fig.suptitle('SGD Variants')

plt.tight_layout()
plt.show()