import os
import time
import numpy as np
import matplotlib.pyplot as plt

from config import CONFIG
from data import *
from plot import *

# Import Problems
from problems.logistics import LogisticRegression
from problems.linear import LinearRegression
from problems.neural import NeuralNet

# Import Optimizers
from optimizers.sgd import SGD
from optimizers.adam import Adam
from optimizers.sag import SAG
from optimizers.svrg import SVRG
from optimizers.adagrad import Adagrad

def run_problem_benchmarks(problem_name):
    print(f"\n================ Running: {problem_name.upper()} ================")
    cfg = CONFIG[problem_name]
    
    # Output Folder
    output_dir = f"outputs/{problem_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Data Gen
    if problem_name != "Linear":
        X, y = generate_gaussian_blobs(n_features=cfg["n_features"], n_samples=cfg["n_samples"])
    else :
        X,y = generate_regression_data(n_features=cfg["n_features"],n_samples=cfg["n_samples"],noise=cfg["noise"])

    # Problems Setup
    if problem_name == "logistic":
        prob_instance = LogisticRegression(X, y)
        w_init = np.random.randn(cfg["n_features"] + 1)
    elif problem_name == "Linear":
        prob_instance = LinearRegression(X, y)
        w_init = np.random.randn(cfg["n_features"] + 1)
        w_s=np.linalg.lstsq(X,y,rcond=None)[0]
        loss_=prob_instance.loss(w_s)
    elif problem_name == "neural_network":
        prob_instance = NeuralNet(X, y,hidden=cfg["hidden"],activation=cfg["activation"])
        w1=np.random.randn(*prob_instance.shape)*0.1
        w2=np.random.randn(*prob_instance.shape_)*0.1
        w_init=prob_instance.flatten(w1,w2)
        
    initial_loss = prob_instance.loss(w_init)
    
    # Optimizers init
    optimizers = {
        "SGD":      SGD(prob_instance, "SGD",w_init),
        "Polyak":   SGD(prob_instance, "polyak",w_init),
        "Nesterov": SGD(prob_instance, "nesterov",w_init),
        "Adagrad":  Adagrad(w_init,prob_instance),
        "Adam":     Adam(w_init,prob_instance),
        "SAG":      SAG(prob_instance,w_init),
        "SVRG":     SVRG(prob_instance),
    }
    

    # ------------------ BENCHMARK 1: ITERATIONS ------------------
    print("Running Iteration Benchmark...")
    history = {name: [initial_loss] for name in optimizers}
    weights = {name: np.copy(w_init) for name in optimizers}
    times = {name: [0.0] for name in optimizers}
    t_cumul = {name: 0.0 for name in optimizers}

    t1=time.perf_counter()

    for i in range(cfg["n_iterations"]):
        indices = np.random.choice(cfg["n_samples"], size=(cfg["num_batch"] if "num_batch" in cfg else cfg["batch_size"],), replace=False)
        for name, opt in optimizers.items():
            opt_params = cfg["optimizers"][name]
            
            t0 = time.perf_counter()
            weights[name] = opt.step(weights[name], indices, i+1, **opt_params)
            t_cumul[name] +=(time.perf_counter()-t0)
            
            history[name].append(prob_instance.loss(weights[name]))
            times[name].append(t_cumul[name])

    print(f"Benchmark 1 lasted {(time.perf_counter()-t1):.3f} seconds")   

    # X_costs
    iterations = np.arange(cfg["n_iterations"] + 1)
    x_costs = {name: iterations * cfg["batch_size"] for name in optimizers}
    x_costs["SAG"] = iterations *1 + cfg["n_samples"]
    x_costs["SVRG"] = iterations*(cfg["n_samples"]*5)

    print("Saving Benchmar 1 Plot Results ...")
    # Plot Saving
    plot_combined(history, x_costs)
    plt.savefig(f"{output_dir}/iterations_combined.png")
    plt.close()
    
    plot_grid(history, x_costs)
    plt.savefig(f"{output_dir}/iterations_grid.png")
    plt.close()
    if problem_name == "Linear":
        plot_log_scale(history, x_costs,problem_name,loss_)
    else :
        plot_log_scale(history,x_costs,problem_name)
    plt.savefig(f"{output_dir}/iterations_log.png")
    plt.close()
    
    if problem_name != "Linear":
        plot_decision_boundary(weights, X, y, LogisticRegression(X,y).sigmoid)
        plt.savefig(f"{output_dir}/iterations_decision.png")
        plt.close()
    
    plot_wallclock_log(history,times)
    plt.savefig(f"{output_dir}/iterations_clocklog.png")
    plt.close()

    plot_epoch_grid(history,x_costs)
    plt.savefig(f"{output_dir}/iterations_epoch.png")
    plt.close()

    plot_wallclock(history, times)
    plt.savefig(f"{output_dir}/iterations_wallclock.png")
    plt.close()
    
    #Optimizers Reset for benchmark 2
    optimizers = {
    "SGD":      SGD(prob_instance, "SGD", w_init),
    "Polyak":   SGD(prob_instance, "polyak", w_init),
    "Nesterov": SGD(prob_instance, "nesterov", w_init),
    "Adagrad":  Adagrad(w_init, prob_instance),
    "Adam":     Adam(w_init, prob_instance),
    "SAG":      SAG(prob_instance, w_init),
    "SVRG":     SVRG(prob_instance),
}
    # ------------------ BENCHMARK 2: GRADIENT CALLS ------------------
    print("Running Gradient Calls Benchmark...")

    history_grad = {name: [initial_loss] for name in optimizers}
    weights_grad = {name: np.copy(w_init) for name in optimizers}
    x_costs_grad = {name: [0] for name in optimizers}

    t1=time.perf_counter()

    for name, opt in optimizers.items():
        current_grads=0
        if (name=="SAG"): current_grads= cfg["n_samples"]
        i =0
        opt_params= cfg["optimizers"][name]
        while current_grads <cfg["max_grad_calls"]:
            i+=1
            indices=np.random.choice(cfg["n_samples"], size=(cfg["batch_size"],), replace=False)
            
            weights_grad[name] =opt.step(weights_grad[name],indices,i, **opt_params)
            if name == "SVRG":
                current_grads += (cfg["n_samples"]*5)
            elif name == "SAG":
                current_grads += 1
            else:
                current_grads += cfg["batch_size"]
                
            history_grad[name].append(prob_instance.loss(weights_grad[name]))
            x_costs_grad[name].append(current_grads)

    print(f"Benchmark lasted {(time.perf_counter()-t1):.3f} seconds")


    print("Plotting Gradient Calls Results")


    plot_combined(history_grad, x_costs_grad)
    plt.savefig(f"{output_dir}/grad_calls_combined.png")
    plt.close()
    
    plot_grid(history_grad, x_costs_grad)
    plt.savefig(f"{output_dir}/grad_calls_grid.png")
    plt.close()
    
    if problem_name== "Linear":
        plot_log_scale(history_grad, x_costs_grad,problem_name,loss_)
    else:
        plot_log_scale(history_grad,x_costs_grad,problem_name)

    plt.savefig(f"{output_dir}/grad_calls_log.png")
    plt.close()

    if problem_name!="Linear":   
        plot_decision_boundary(weights_grad, X, y, LogisticRegression(X,y).sigmoid)
        plt.savefig(f"{output_dir}/grad_calls_decision.png")
        plt.close() 
    
    print(f"Plots saved to: {output_dir}/")

if __name__ == "__main__":
    np.random.seed(5)
    for problem in ["Linear"]:
        print(f"Starting problem {problem}")
        run_problem_benchmarks(problem)