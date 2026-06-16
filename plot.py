import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

COLORS = {
    "SGD":              "#6C8EBF",
    "Polyak":           "#EF9F27",
    "Nesterov":         "#D85A30",
    "Adagrad":          "#7F77DD",
    "Adam":             "#1D9E75",
    "SAG":              "#C45BAA",
    "SVRG":             "#E8E8E8",
    "NN":               "#F4D03F",
}

STYLE = {
    "bg":       "#0F1117",
    "panel":    "#1A1D27",
    "border":   "#2A2D3A",
    "text":     "#E0E0E0",
    "subtext":  "#8A8FA8",
    "grid":     "#1E2130",
}

def _apply_dark(fig, axes):
    fig.patch.set_facecolor(STYLE["bg"])
    for ax in np.array(axes).flat:
        ax.set_facecolor(STYLE["panel"])
        ax.tick_params(colors=STYLE["subtext"], labelsize=8)
        ax.xaxis.label.set_color(STYLE["subtext"])
        ax.yaxis.label.set_color(STYLE["subtext"])
        ax.title.set_color(STYLE["text"])
        for spine in ax.spines.values():
            spine.set_edgecolor(STYLE["border"])
        ax.grid(True, linestyle="--", linewidth=0.4, color=STYLE["grid"])


def plot_grid(history, x_costs, title="SGD Variants"):
    names = list(history.keys())
    n =len(names)
    cols= 3
    rows=(n+cols-1)//cols

    fig, axs = plt.subplots(rows, cols,figsize=(8,6),layout="constrained")
    _apply_dark(fig, axs)
    fig.suptitle(title, color=STYLE["text"], fontsize=13, fontweight="bold")

    for idx, name in enumerate(names):
        ax = axs.flat[idx]
        color = COLORS.get(name, "#FFFFFF")
        ax.plot(x_costs[name], history[name], color=color, linewidth=1.6)
        ax.set_title(name, fontsize=10, color=color)
        ax.set_xlabel("Gradient calls", fontsize=8)
        ax.set_ylabel("Loss", fontsize=8)

    for idx in range(len(names), rows * cols):
        axs.flat[idx].set_visible(False)

    #plt.tight_layout()

def plot_epoch_grid(history, x_costs, title="SGD Variants"):
    names = list(history.keys())

    fig, axs = plt.subplots(2, 2,figsize=(8,6),layout="constrained")
    _apply_dark(fig, axs)
    fig.suptitle(title, color=STYLE["text"], fontsize=13, fontweight="bold")

    for idx, name in enumerate(names):

        color = COLORS.get(name, "#FFFFFF")

        axs[0,0].plot(x_costs[name], history[name], color=color, linewidth=1.6, label=name)
        axs[0,0].set_xlabel("Gradient Calls")
        axs[0,0].set_ylabel("Loss", fontsize=8)
        legend = axs[0,0].legend(loc="upper right", framealpha=0.15,
                       labelcolor="linecolor", fontsize=9)
        
        axs[0,1].plot(x_costs[name],history[name], color=color, linewidth=1.6, label=name)
        axs[0,1].set_xlabel("Gradient Calls (Log)")
        axs[0,1].set_xscale('symlog')
        axs[0,1].set_yscale('symlog')
        axs[0,1].set_ylabel("Loss (Log)", fontsize=8)
        legend = axs[0,1].legend(loc="upper right", framealpha=0.15,
                       labelcolor="linecolor", fontsize=9)
        
        axs[1,0].plot(history[name], color=color, linewidth=1.6, label=name)
        axs[1,0].set_xlabel("Iterations")
        axs[1,0].set_ylabel("Loss", fontsize=8)
        legend = axs[1,0].legend(loc="upper right", framealpha=0.15,
                       labelcolor="linecolor", fontsize=9)
        
        axs[1,1].plot(history[name], color=color, linewidth=1.6, label=name)
        axs[1,1].set_xlabel("Iterations")
        axs[1,1].set_yscale('symlog')
        axs[1,1].set_ylabel("Loss (Log)", fontsize=8)
        legend = axs[1,1].legend(loc="upper right", framealpha=0.15,
                       labelcolor="linecolor", fontsize=9)
        legend.get_frame().set_facecolor(STYLE["panel"])

        

    for idx in range(len(names), 4):
        axs.flat[idx].set_visible(False)


def plot_combined(history, x_costs):
    fig, ax = plt.subplots()
    _apply_dark(fig, [ax])
    ax.set_title("Stochastic Optimization — Convergence", color=STYLE["text"], fontsize=12)

    for name, losses in history.items():
        lw = 2.5 if name == "SVRG" else 1.5
        ax.plot(x_costs[name], losses, label=name,
                color=COLORS.get(name, "#FFFFFF"), linewidth=lw, alpha=0.9)
        #ax.set_xlim(min(x_costs[name]))

    ax.set_xlabel("Gradient calls", fontsize=9)
    ax.set_xscale('symlog')
    ax.set_ylabel("Loss", fontsize=9)
    legend = ax.legend(loc="upper right", framealpha=0.15,
                       labelcolor="linecolor", fontsize=9)
    legend.get_frame().set_facecolor(STYLE["panel"])
    plt.tight_layout()





def plot_combined_grad(history, x_costs):
    fig, ax = plt.subplots()
    _apply_dark(fig, [ax])
    ax.set_title("Stochastic Optimization — Convergence", color=STYLE["text"], fontsize=12)

    for name, losses in history.items():
        lw = 2.5 if name == "SVRG" else 1.5
        ax.plot(x_costs[name], losses, label=name,
                color=COLORS.get(name, "#FFFFFF"), linewidth=lw, alpha=0.9)
        #ax.set_xlim(min(x_costs[name]))

    ax.set_xlabel("Gradient calls", fontsize=9)
    ax.set_ylabel("Loss", fontsize=9)
    legend = ax.legend(loc="upper right", framealpha=0.15,
                       labelcolor="linecolor", fontsize=9)
    legend.get_frame().set_facecolor(STYLE["panel"])
    plt.tight_layout()

def plot_log_scale(history, x_costs, problem,loss_=None):
    fig, ax = plt.subplots()
    _apply_dark(fig, [ax])
    ax.set_title("Suboptimality — Log Scale",
                 color=STYLE["text"], fontsize=12)
    
    if problem == "Linear":
        loss = loss_
    else :
         loss = min(min(v) for v in history.values())
    for name, losses in history.items():
        lw = 2.5 if name == "SVRG" else 1.5
        subopt = [l-loss + 1e-10 for l in losses]
        ax.semilogy(x_costs[name], subopt, label=name,
                    color=COLORS.get(name, "#FFFFFF"), linewidth=lw, alpha=0.9, nonpositive='mask')
    ax.set_xlabel("Gradient calls", fontsize=9)
    #ax.set_xscale('symlog')
    legend = ax.legend(loc="upper right", framealpha=0.15,
                       labelcolor="linecolor", fontsize=9)
    legend.get_frame().set_facecolor(STYLE["panel"])
    plt.tight_layout()


def plot_decision_boundary(weights, X, y, sigmoid_fn,problem="logistic",prob_instance=None):
    """
    weights   : dict  name → w  (shape d+1 with bias prepended)
    X         : (n, d) no bias column
    y         : (n,)
    sigmoid_fn: callable σ(z) → probabilities
    """
    names = list(weights.keys())
    n =len(names)
    cols=3
    rows =(n+cols-1)//cols

    x_min, x_max=X[:, 1].min()-0.5,X[:, 1].max() +0.5
    y_min, y_max =X[:, 2].min()-0.5, X[:, 2].max()+0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),np.linspace(y_min, y_max, 300))
    grid = np.c_[np.ones(xx.ravel().shape), xx.ravel(), yy.ravel()]
    fig, axs = plt.subplots(rows, cols)
    _apply_dark(fig, axs)
    fig.suptitle("Decision Boundaries", color=STYLE["text"], fontsize=13,fontweight="bold")

    cmap_bg  = ListedColormap(["#1A2A3A", "#1A3A2A"])
    cmap_pts = ListedColormap(["#6C8EBF", "#EF9F27"])

    for idx, name in enumerate(names):
        ax = axs.flat[idx]
        w = weights[name]
        if problem!="neural_network":
            Z=sigmoid_fn(grid@w).reshape(xx.shape)
        else :
            Z=prob_instance.predict_grid(grid,weights[name]).reshape(xx.shape)

        ax.contourf(xx, yy, Z, alpha=0.35, cmap=cmap_bg, levels=50)
        ax.contour(xx, yy, Z, levels=[0.5],
                   colors=[COLORS.get(name, "#FFFFFF")], linewidths=1.5)
        ax.scatter(X[:,1], X[:,2], c=y, cmap=cmap_pts,
                   s=18, edgecolors="none", alpha=0.8)
        ax.set_title(name, fontsize=10, color=COLORS.get(name, "#FFFFFF"))
        ax.set_xticks([])
        ax.set_yticks([])

    for idx in range(len(names), rows*cols):
        axs.flat[idx].set_visible(False)
    plt.tight_layout()


def plot_grad_norms(grad_histories):
    fig, ax = plt.subplots()
    _apply_dark(fig, [ax])
    ax.set_title("Gradient Norm vs Iterations", color=STYLE["text"], fontsize=12)

    for name, norms in grad_histories.items():
        ax.plot(norms, label=name, color=COLORS.get(name, "#FFFFFF"),
                linewidth=1.4, alpha=0.85)

    ax.set_xlabel("Iterations", fontsize=9)
    ax.set_ylabel("‖∇L‖", fontsize=9)
    legend = ax.legend(loc="upper right", framealpha=0.15,
                       labelcolor="linecolor", fontsize=9)
    legend.get_frame().set_facecolor(STYLE["panel"])
    plt.tight_layout()


def plot_wallclock_log(history, times):
    """
    times : dict  name → list of cumulative seconds
    """
    fig, ax = plt.subplots()
    _apply_dark(fig, [ax])
    ax.set_title("Convergence vs Wall-Clock Time", color=STYLE["text"], fontsize=12)

    for name, losses in history.items():
        lw = 2.5 if name == "SVRG" else 1.5 #highlight svrg
        ax.plot(times[name], losses, label=name,
                color=COLORS.get(name, "#FFFFFF"), linewidth=lw, alpha=0.9)
    ax.set_xlabel("Time (Log)", fontsize=9)
    ax.set_xscale('symlog')
    ax.set_ylabel("Loss", fontsize=9)
    legend = ax.legend(loc="upper right", framealpha=0.15,
                       labelcolor="linecolor", fontsize=9)
    legend.get_frame().set_facecolor(STYLE["panel"])
    plt.tight_layout()


def plot_wallclock(history, times):
    """
    times : dict  name → list of cumulative seconds
    """
    fig, ax = plt.subplots(1,2)
    _apply_dark(fig, [ax])
    fig.suptitle("Convergence vs Wall-Clock Time", color=STYLE["text"], fontsize=12)

    for name, losses in history.items():
        lw = 2.5 if name == "SVRG" else 1.5 #highlight svrg
        ax[0].plot(times[name], losses, label=name,
                color=COLORS.get(name, "#FFFFFF"), linewidth=lw, alpha=0.9)
        ax[1].plot(times[name], losses, label=name,
                color=COLORS.get(name, "#FFFFFF"), linewidth=lw, alpha=0.9)
    
    ax[0].set_xlabel("Time (seconds)", fontsize=9)
    ax[0].set_ylabel("Loss", fontsize=9)
    legend = ax[0].legend(loc="upper right", framealpha=0.15,
                       labelcolor="linecolor", fontsize=9)

    ax[1].set_xlim(0,max(times["Nesterov"]))
    ax[1].set_xlabel("Time (seconds)", fontsize=9)
    ax[1].set_ylabel("Loss", fontsize=9)
    legend = ax[1].legend(loc="upper right", framealpha=0.15,
                       labelcolor="linecolor", fontsize=9)
    legend.get_frame().set_facecolor(STYLE["panel"])

    plt.tight_layout()

