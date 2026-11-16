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

# Step 2 - fit_gda
def fit_gda(X, y, C):
    """
    Fit a Gaussian Discriminant Analysis (GDA) model using
    Maximum Likelihood Estimation (MLE).

    Parameters
    ----------
    X : ndarray, shape (N, D)
        Training data containing N observations and D features.
    y : ndarray, shape (N,)
        Integer class labels ranging from 0 to C-1.
    C : int
        Number of classes.

    Returns
    -------
    dict
        A dictionary containing:
        - 'pi': Class prior probabilities, shape (C,).
        - 'mu': Class mean vectors, shape (C, D).
        - 'Sigma': Class covariance matrices, shape (C, D, D).
    """

    # Convert inputs to NumPy arrays.
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)

    # Determine the number of observations and features.
    N, D = X.shape

    # Initialize arrays for class priors, means, and covariances.
    pi = np.zeros(C)
    mu = np.zeros((C, D))
    Sigma = np.zeros((C, D, D))

    # Estimate the Gaussian parameters independently for each class.
    for c in range(C):

        # Select all observations belonging to class c.
        X_c = X[y == c]

        # Count the number of observations in the current class.
        N_c = X_c.shape[0]

        # Every class must contain at least one observation.
        if N_c == 0:
            raise ValueError(f"Class {c} has no training examples.")

        # Estimate the class prior probability using MLE.
        pi[c] = N_c / N

        # Estimate the class mean vector.
        mu[c] = np.mean(X_c, axis=0)

        # Center the observations around their class mean.
        centered_X = X_c - mu[c]

        # Estimate the class covariance matrix using MLE.
        # Divide by N_c rather than N_c - 1.
        Sigma[c] = (centered_X.T @ centered_X) / N_c

    # Return all estimated Gaussian parameters.
    return {
        "pi": pi,
        "mu": mu,
        "Sigma": Sigma
    }

# Step 3 - gda_log_joint
def gda_log_joint(params, X):
    """
    Evaluate the GDA discriminant function for every observation
    and every class.

    Parameters
    ----------
    params : dict
        Dictionary containing:
        - 'pi': Class prior probabilities, shape (C,).
        - 'mu': Class mean vectors, shape (C, D).
        - 'Sigma': Class covariance matrices, shape (C, D, D).

    X : ndarray, shape (N, D)
        Input observations.

    Returns
    -------
    log_joint : ndarray, shape (N, C)
        Unnormalized log-posterior scores for each observation
        and each class.
    """

    # Extract the Gaussian parameters from the fitted GDA model.
    pi = params["pi"]
    mu = params["mu"]
    Sigma = params["Sigma"]

    # Determine the number of classes.
    C = len(pi)

    # Initialize the output matrix for log-joint probabilities.
    N = X.shape[0] if X.ndim > 1 else 1
    log_joint = np.zeros((N, C))

    # Calculate the log-joint probability for each class.
    for c in range(C):

        # Combine the log class prior with the Gaussian log-density.
        log_joint[:, c] = (
            np.log(pi[c])
            + mvn_logpdf(X, mu[c], Sigma[c])
        )

    return log_joint

