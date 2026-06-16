import numpy as np

def generate_gaussian_blobs(n_features=2, n_samples=1000, random_state_seed=5,factor=0.5,center=0.5):
    """
    Generates linearly separable Gaussian blobs for binary classification.
    Returns:
        X: array of shape (n_samples, n_features + 1) -> includes bias column
        y: array of shape (n_samples,) with labels 0 or 1
    """

    np.random.seed(random_state_seed)
    n_class=int(n_samples*factor)

    b=np.array([1]*n_samples)
    X0=np.random.randn(n_class,n_features)-center
    y0=np.zeros(n_class)

    X1=np.random.randn(n_samples-n_class,n_features)+center
    y1=np.ones(n_samples-n_class)

    X = np.vstack((X0, X1))
    y = np.concatenate((y0, y1))
    X = np.insert(X, 0, b, axis=1)
    
    indices = np.random.permutation(n_samples) #shuffling points
    X = X[indices]
    y = y[indices]

    return X,y

def generate_regression_data(n_features=2,n_samples=1000,noise=1e-1,random_state_seed=5):

    np.random.seed(random_state_seed)

    X=np.random.randn(n_samples,n_features)
    b=np.array([1]*n_samples)
    X=np.insert(X,0,b,axis=1)
    
    w=np.random.randn(n_features+1)
    y= X@w + noise*np.random.randn(n_samples)

    return X,y
