import numpy as np
import matplotlib.pyplot as plt
from data import*
from problems.logistics import*
from optimizers.sgd import *
from optimizers.adagrad import*
from optimizers.adam import *
from optimizers.sag import *
from optimizers.svrg import *
from problems.linear import*
from problems.neural import*
from plot import*
import time

np.random.seed(5)

#parameters
n_samples=1000
n_features=2
alpha=0.01
n_iterations=150
batch_size=5
beta=0.9

#data generation
X, y=generate_gaussian_blobs(n_features,n_samples,center=0.75)

L=LogisticRegression(X,y)
Lin=LinearRegression(X,y)
nn=NN(X,y)

w= np.random.randn(n_features + 1)


w0 = np.copy(w) # SGD
w1 = np.copy(w) # Adagrad
w2 = np.copy(w) # Polyak
w3 = np.copy(w) # Nesterov
w4 = np.copy(w) # Adam
w5 = np.copy(w) # SAG
w6 = np.copy(w) # SVRG

#storing for plot
history = {
    "SGD":      [],
    "Polyak":   [],
    "Nesterov": [],
    "Adagrad":  [],
    "Adam":     [],
    "SAG":      [],
    "SVRG":     [],
}

history_Linear_Regression={
    "SGD":      [],
    "Polyak":   [],
    "Nesterov": [],
    "Adagrad":  [],
    "Adam":     [],
    "SAG":      [],
    "SVRG":     [],
}

history_NN={
    "SGD":      [],
    "Polyak":   [],
    "Nesterov": [],
    "Adagrad":  [],
    "Adam":     [],
    "SAG":      [],
    "SVRG":     [],
}

weights = {
    "SGD": w0, "Polyak": w2, "Nesterov": w3,
    "Adagrad": w1, "Adam": w4, "SAG": w5, "SVRG": w6
}


optimizers={
    "SGD":      SGD(X, y, w,"SGD"),
    "Polyak":   SGD(X, y, w,"polyak"),
    "Nesterov": SGD(X, y, w,"nesterov"),
    "Adagrad":  Adagrad(X, y, w),
    "Adam":     Adam(X, y, w),
    "SAG":      SAG(X, y, w),
    "SVRG":     SVRG(X, y, w),
}

optimizers_Linear={
    "SGD":      SGD(X, y, w,"SGD"),
    "Polyak":   SGD(X, y, w,"polyak"),
    "Nesterov": SGD(X, y, w,"nesterov"),
    "Adagrad":  Adagrad(X, y, w),
    "Adam":     Adam(X, y, w),
    "SAG":      SAG(X, y, w),
    "SVRG":     SVRG(X, y, w),
}

optimizers_NN={
    "SGD":      SGD(X, y, w,"SGD"),
    "Polyak":   SGD(X, y, w,"polyak"),
    "Nesterov": SGD(X, y, w,"nesterov"),
    "Adagrad":  Adagrad(X, y, w),
    "Adam":     Adam(X, y, w),
    "SAG":      SAG(X, y, w),
    "SVRG":     SVRG(X, y, w),
}

times = {name: [] for name in history}
t_cumul = {name: 0.0 for name in history}

#added different steps for x axis to account for different computation approaches and hold a fair comparaison on the plots

iterations = np.arange(n_iterations)

x_sgd =iterations *batch_size 
x_polyak =iterations* batch_size
x_adam = iterations *batch_size
x_adagrad= iterations * batch_size
x_sag= iterations *batch_size
x_svrg=iterations*(n_samples+(2*n_samples))

#benchmarking loop
for i in range(n_iterations):
    indices=np.random.choice(n_samples,size=(batch_size,),replace=False)
    
    for name, optimizer in optimizers.items():
        t0 = time.perf_counter()
        history[name].append(L.loss(weights[name]))
        weights[name]=optimizer.step(weights[name],indices,i+1)
        t_cumul[name] += time.perf_counter()-t0
        times[name].append(t_cumul[name])
        #history_Linear[name].append()




#plotting
print (" Plotting ")
x_costs = {
    "SGD":      x_sgd,
    "Polyak":   x_polyak,
    "Nesterov": x_sgd,
    "Adagrad":  x_adagrad,
    "Adam":     x_adam,
    "SAG":      x_sag,
    "SVRG":     x_svrg,
}

plot_grid(history, x_costs)
plot_combined(history, x_costs)
plot_log_scale(history, x_costs)
plot_decision_boundary(weights, X, y, L.sigmoid)
plot_wallclock(history,times)
plot_wallclock_log(history,times)
plot_epoch_grid(history,x_costs)
plt.show()