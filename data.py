import numpy as np

def generate_gaussian_blobs(n_features=2, n_samples=1000, random_state_seed=5):
    """
    Generates linearly separable Gaussian blobs for binary classification.
    Returns:
        X: array of shape (n_samples, n_features + 1) -> includes bias column
        y: array of shape (n_samples,) with labels 0 or 1
    """

    np.random.seed(random_state_seed)
    b=np.array([1]*n_samples)
    X=np.random.randn(n_samples,n_features)
    X=np.insert(X,0,b,axis=1)
    y=np.random.randint(0,2,np.size(n_samples,))

    return X,y

X , y =generate_gaussian_blobs()
print(np.shape(X))
print(np.shape(y))
