import numpy as np
import matplotlib.pyplot as plt
from data import*
from problems.logistics import*

#parameters
n_samples=1000
n_features=2
alpha=0.1
n_iterations=500
batch_size=10

#data generation
X, y=generate_gaussian_blobs(n_features,n_samples)
np.random.seed(5)
L=LogisticRegression(X,y)
w=np.random.randn(n_features+1)

#storing for plot
history=[]

#benchmarking loop
for i in range(n_iterations):
    indices=np.random.choice(n_samples,size=(batch_size,),replace=False)
    grad=L.gradient(w,indices)
    w=w-alpha*grad
    history.append(L.loss(w))


#plotting
plt.plot(history)
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("LR Loss vs Iterations")
plt.show()
