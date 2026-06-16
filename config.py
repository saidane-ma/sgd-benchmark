CONFIG = {
    "logistic": {
        "n_samples": 1000,
        "n_features": 2,
        "n_iterations": 500,
        "batch_size": 50,
        "max_grad_calls": 150000,
        "optimizers": {
            "SGD":      {"alpha": 0.01},
            "Polyak":   {"alpha": 0.01, "beta": 0.9},
            "Nesterov": {"alpha": 0.001, "beta": 0.99},
            "Adagrad":  {"alpha": 0.5},
            "Adam":     {"alpha": 0.05, "beta1": 0.9, "beta2": 0.999},
            "SAG":      {"alpha": 0.001},
            "SVRG":     {"alpha": 0.01}
        }
    },
    "Linear": {
        "n_samples": 1000,
        "n_features": 2,
        "n_iterations": 200,
        "batch_size": 32,
        "noise": 0.1,
        "max_grad_calls": 50000,
        "optimizers": {
            "SGD":      {"alpha": 0.05},
            "Polyak":   {"alpha": 0.005, "beta": 0.9},
            "Nesterov": {"alpha": 0.0001, "beta": 0.5},
            "Adagrad":  {"alpha": 0.5},
            "Adam":     {"alpha": 0.2,  "beta1": 0.9, "beta2": 0.999},
            "SAG":      {"alpha": 0.001},
            "SVRG":     {"alpha": 0.005}
        }
    },
    "neural_network": {
        "n_samples": 500,
        "n_features": 2,
        "hidden": 4,
        "activation": "sigmoid",
        "n_iterations": 300,
        "batch_size": 10,
        "max_grad_calls": 200000,
        "optimizers": {
            "SGD":      {"alpha": 0.1},
            "Polyak":   {"alpha": 0.05, "beta": 0.9},
            "Nesterov": {"alpha": 0.05, "beta": 0.9},
            "Adagrad":  {"alpha": 0.01},
            "Adam":     {"alpha": 0.001,  "beta1": 0.9, "beta2": 0.999},
            "SAG":      {"alpha": 0.01},
            "SVRG":     {"alpha": 0.005}
        }
    }
}