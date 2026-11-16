"""
Generative Classifiers from Scratch: GDA, Naive Bayes and Fisher LDA vs Logistic Regression

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - mvn_logpdf
import numpy as np

def mvn_logpdf(X, mu, Sigma):
    """
    Compute the log probability density of a multivariate Gaussian
    distribution for each observation in X.

    Parameters
    ----------
    X : ndarray, shape (N, D) or (D,)
        Observations. A single observation of shape (D,) is treated
        as one row.
    mu : ndarray, shape (D,)
        Mean vector of the Gaussian distribution.
    Sigma : ndarray, shape (D, D)
        Covariance matrix of the Gaussian distribution.

    Returns
    -------
    log_pdf : ndarray, shape (N,)
        Log-density evaluated at each observation.
    """

    # Convert inputs to floating-point NumPy arrays.
    X = np.asarray(X, dtype=float)
    mu = np.asarray(mu, dtype=float)
    Sigma = np.asarray(Sigma, dtype=float)

    # Treat a single observation as a batch containing one row.
    if X.ndim == 1:
        X = X.reshape(1, -1)

    # Determine the number of features.
    D = mu.shape[0]

    # Cholesky decomposition: Sigma = L @ L.T.
    # Raises np.linalg.LinAlgError if Sigma is not positive definite.
    L = np.linalg.cholesky(Sigma)

    # Center each observation by subtracting the mean vector.
    centered_X = X - mu

    # Solve L @ Z.T = centered_X.T using a triangular system.
    # This avoids explicitly calculating the inverse of Sigma.
    Z = np.linalg.solve(L, centered_X.T).T

    # Calculate the log determinant using the Cholesky factor.
    # log|Sigma| = 2 * sum(log(diag(L))).
    log_det = 2.0 * np.sum(np.log(np.diag(L)))

    # Calculate the squared Mahalanobis distances.
    # (x - mu).T @ Sigma^(-1) @ (x - mu) = ||Z||^2.
    mahalanobis_sq = np.sum(Z ** 2, axis=1)

    # Compute the multivariate Gaussian log-density.
    log_pdf = -0.5 * (
        D * np.log(2.0 * np.pi)
        + log_det
        + mahalanobis_sq
    )

    return np.asarray(log_pdf, dtype=float)

