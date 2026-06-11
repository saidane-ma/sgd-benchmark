# SGD Benchmark Suite

A modular Python benchmarking framework to evaluate and compare the convergence properties of various Stochastic Gradient Descent (SGD) variants against classic and variance-reduced optimization algorithms.

---

## 📊 Project Structure

```text
sgd-benchmark/
├── optimizers/
│   ├── base.py          # Abstract base class for optimizers
│   ├── sgd.py           # Vanilla SGD + momentum + Nesterov
│   ├── adagrad.py       # AdaGrad optimizer
│   ├── adam.py          # Adam optimizer
│   ├── sag.py           # Stochastic Average Gradient (SAG)
│   └── svrg.py          # Stochastic Variance Reduced Gradient (SVRG)
├── problems/
│   ├── logistic.py      # Logistic regression loss & gradient
│   ├── linear.py        # Linear regression loss & gradient
│   └── neural.py        # Neural network loss & gradient
├── data.py              # Synthetic dataset generators
├── benchmark.py         # Main entry point to run experiments
├── plot.py              # Script to generate convergence figures
└── README.md            # Project documentation
```
🚀 Getting Started
Prerequisites
Ensure you have Python 3.8+ installed. This project relies on standard scientific computing libraries:

NumPy

SciPy

Matplotlib

Installation
Clone the repository:

```
git clone [https://github.com/](https://github.com/)saidane-ma/sgd-benchmark.git
   cd sgd-benchmark
```

🛠️ Usage
1. Running the Benchmark
To run the full suite (all optimizers across all optimization problems), execute the main benchmark script:

```
python benchmark.py
```

2. Plotting Results
To generate and view the convergence plots (e.g., Optimality Gap / Loss vs. Iterations):

```
python plot.py
```

🔬 Algorithms & Optimization Problems
Supported Optimizers
Standard / Adaptive: Vanilla SGD, Momentum, Nesterov Accelerated Gradient, AdaGrad, Adam.

Variance-Reduced: Stochastic Average Gradient (SAG), Stochastic Variance Reduced Gradient (SVRG).

Problems Evaluated
Linear Regression: Convex, quadratic objective.

Logistic Regression: Strongly convex, smooth objective.

Neural Networks: Non-convex optimization problem.

📜 License
This project is licensed under the MIT License - see the LICENSE file for details.

✍️ Author
Mohamed Amine SAIDANE - Initial Work - @saidane-ma
