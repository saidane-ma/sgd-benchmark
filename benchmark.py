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
batch_size=15
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

initial_loss=L.loss(w)

#storing for plot
history = {
    "SGD":      [initial_loss],
    "Polyak":   [initial_loss],
    "Nesterov": [initial_loss],
    "Adagrad":  [initial_loss],
    "Adam":     [initial_loss],
    "SAG":      [initial_loss],
    "SVRG":     [initial_loss],
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


times = {name: [0] for name in history}
t_cumul = {name: 0.0 for name in history}

#added different steps for x axis to account for different computation approaches and hold a fair comparaison on the plots

iterations = np.arange(n_iterations+1)

x_sgd =iterations *batch_size 
x_polyak =iterations* batch_size
x_adam = iterations *batch_size
x_adagrad= iterations * batch_size
x_sag= iterations *1
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

x_costs = {
    "SGD":      x_sgd,
    "Polyak":   x_polyak,
    "Nesterov": x_sgd,
    "Adagrad":  x_adagrad,
    "Adam":     x_adam,
    "SAG":      x_sag,
    "SVRG":     x_svrg,
}


#plotting
print (" Plotting Iteration Benchmarking")

plot_grid(history, x_costs)
plot_combined(history, x_costs)
plot_log_scale(history, x_costs)
plot_decision_boundary(weights, X, y, L.sigmoid)
plot_wallclock(history,times)
plot_wallclock_log(history,times)
plot_epoch_grid(history,x_costs)



#//////////////////////////////////////////////////////////


history={
    "SGD":      [initial_loss],
    "Polyak":   [initial_loss],
    "Nesterov": [initial_loss],
    "Adagrad":  [initial_loss],
    "Adam":     [initial_loss],
    "SAG":      [initial_loss],
    "SVRG":     [initial_loss],
}


print("Grad Calls Based Benchmarking ...")

#benchmark loop with max grads
max_grad_calls = 100000

w0 = np.copy(w) # SGD
w1 = np.copy(w) # Adagrad
w2 = np.copy(w) # Polyak
w3 = np.copy(w) # Nesterov
w4 = np.copy(w) # Adam
w5 = np.copy(w) # SAG
w6 = np.copy(w) # SVRG

weights = {
    "SGD": w0, "Polyak": w2, "Nesterov": w3,
    "Adagrad": w1, "Adam": w4, "SAG": w5, "SVRG": w6
}


optimizers={
    "SGD":      SGD(X, y, weights["SGD"],"SGD"),
    "Polyak":   SGD(X, y, weights["Polyak"],"polyak"),
    "Nesterov": SGD(X, y, weights["Nesterov"],"nesterov"),
    "Adagrad":  Adagrad(X, y, weights["Adagrad"]),
    "Adam":     Adam(X, y, weights["Adam"]),
    "SAG":      SAG(X, y, weights["SAG"]),
    "SVRG":     SVRG(X, y, weights["SVRG"]),
}

times = {name: [0] for name in history}
t_cumul = {name: 0.0 for name in history}


x_costs = {
    "SGD":      [0],
    "Polyak":   [0],
    "Nesterov": [0],
    "Adagrad":  [0],
    "Adam":     [0],
    "SAG":      [0],
    "SVRG":     [0],
}


for name, optimizer in optimizers.items():
    t_cumul=0.0
    current_grads=0
    i=0
    while current_grads<max_grad_calls:
        i+=1 
        indices = np.random.choice(n_samples, size=(batch_size,), replace=False)
        
        t0 = time.perf_counter()
        weights[name] = optimizer.step(weights[name], indices,i)
        t_cumul += (time.perf_counter()-t0)
        
        if name == "SVRG":
            current_grads +=(n_samples + 2*n_samples)
        elif name == "SAG":
            current_grads+=1
        else:
            current_grads += batch_size
        history[name].append(L.loss(weights[name]))
        x_costs[name].append(current_grads)
        times[name].append(t_cumul)




#plotting
print (" Plotting Max Grad Benchmarking")

plot_grid(history, x_costs)
plot_combined(history, x_costs)
plot_log_scale(history, x_costs)
plot_decision_boundary(weights, X, y, L.sigmoid)
plot_wallclock(history,times)
plot_wallclock_log(history,times)

plt.show()