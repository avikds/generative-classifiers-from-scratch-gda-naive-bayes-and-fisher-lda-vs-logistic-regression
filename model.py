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

# Step 4 - log_normalize
def log_normalize(L):
    """
    Convert unnormalized log scores into normalized log probabilities
    using the numerically stable Log-Sum-Exp trick.

    Parameters
    ----------
    L : ndarray, shape (N, C)
        Unnormalized log scores for N observations across C classes.

    Returns
    -------
    log_probs : ndarray, shape (N, C)
        Normalized log probabilities, where the probabilities
        in each row sum to 1 after exponentiation.
    """

    # Convert input to a floating-point NumPy array.
    L = np.asarray(L, dtype=float)

    # Find the maximum log score in each row.
    # keepdims=True preserves the shape for broadcasting.
    row_max = np.max(L, axis=1, keepdims=True)

    # Subtract the row maximum before exponentiation
    # to prevent numerical overflow.
    shifted_L = L - row_max

    # Calculate the log-sum-exp for each row.
    log_sum_exp = row_max + np.log(
        np.sum(np.exp(shifted_L), axis=1, keepdims=True)
    )

    # Subtract the log-sum-exp to obtain normalized log probabilities.
    log_probs = L - log_sum_exp

    return log_probs

# Step 5 - sample_gda
def sample_gda(params, n_per_class, rng):
    """
    Generate labeled synthetic examples from a fitted GDA model.

    Parameters
    ----------
    params : dict
        Fitted GDA parameters containing:
        - 'mu': Class mean vectors, shape (C, D).
        - 'Sigma': Class covariance matrices, shape (C, D, D).

    n_per_class : int
        Number of synthetic observations to generate for each class.

    rng : np.random.Generator
        NumPy random number generator used for reproducible sampling.

    Returns
    -------
    X : ndarray, shape (C * n_per_class, D)
        Generated observations, grouped by class.

    y : ndarray, shape (C * n_per_class,)
        Integer class labels corresponding to the generated observations.
    """

    # Extract the class means and covariance matrices.
    mu = params["mu"]
    Sigma = params["Sigma"]

    # Determine the number of classes and features.
    C, D = mu.shape

    # Store generated observations and labels for each class.
    X_samples = []
    y_samples = []

    # Generate samples class by class in ascending class order.
    for c in range(C):

        # Compute the Cholesky factor of the class covariance matrix.
        # Sigma[c] = L @ L.T
        L = np.linalg.cholesky(Sigma[c])

        # Generate standard normally distributed random observations.
        # Each row represents one observation with D independent features.
        Z = rng.standard_normal((n_per_class, D))

        # Transform standard normal samples into the desired Gaussian.
        # The resulting covariance matrix is L @ L.T = Sigma[c].
        X_c = mu[c] + Z @ L.T

        # Assign the current class label to every generated observation.
        y_c = np.full(n_per_class, c, dtype=int)

        # Store the generated observations and their corresponding labels.
        X_samples.append(X_c)
        y_samples.append(y_c)

    # Combine all class-specific observations into a single dataset.
    X = np.vstack(X_samples)

    # Combine all class labels into a single integer array.
    y = np.concatenate(y_samples)

    return X, y

# Step 6 - constrain_covariances
def constrain_covariances(params, kind="full", lam=0.0):
    """
    Replace the class covariance matrices of a fitted GDA model
    with tied, diagonal, or regularized covariance estimates.

    Parameters
    ----------
    params : dict
        Fitted GDA parameters containing:
        - 'pi': Class prior probabilities, shape (C,).
        - 'mu': Class mean vectors, shape (C, D).
        - 'Sigma': Class covariance matrices, shape (C, D, D).

    kind : str, default="full"
        Covariance estimation strategy:
        - "full": Retain individual full covariance matrices.
        - "tied": Use a shared pooled covariance matrix.
        - "diag": Retain only diagonal covariance elements.
        - "tied_diag": Use a shared pooled diagonal covariance matrix.

    lam : float, default=0.0
        Regularization parameter between 0 and 1.
        Controls the amount of diagonal regularization.

    Returns
    -------
    dict
        A new dictionary containing copies of:
        - 'pi': Class prior probabilities.
        - 'mu': Class mean vectors.
        - 'Sigma': Updated covariance matrices.
    """

    # Validate the covariance estimation strategy.
    valid_kinds = {"full", "tied", "diag", "tied_diag"}

    if kind not in valid_kinds:
        raise ValueError(f"Unknown covariance kind: {kind}")

    # Validate the regularization parameter.
    if not np.isscalar(lam):
        raise ValueError("lam must be a scalar between 0 and 1.")

    try:
        lam = float(lam)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("lam must be a scalar between 0 and 1.")

    if not np.isfinite(lam) or not 0.0 <= lam <= 1.0:
        raise ValueError("lam must be between 0 and 1.")

    # Create independent copies of the original model parameters.
    pi = np.array(params["pi"], copy=True)
    mu = np.array(params["mu"], copy=True)
    Sigma = np.array(params["Sigma"], dtype=float, copy=True)

    # Determine the number of classes and feature dimensions.
    C, D, _ = Sigma.shape

    # Apply covariance pooling for tied covariance strategies.
    if kind in {"tied", "tied_diag"}:

        # Compute the prior-weighted pooled covariance matrix.
        # The MLE class priors make this equivalent to the
        # maximum-likelihood tied covariance estimator.
        pooled_cov = np.sum(
            pi[:, None, None] * Sigma,
            axis=0
        )

        # Assign the same pooled covariance to every class.
        Sigma = np.repeat(
            pooled_cov[np.newaxis, :, :],
            C,
            axis=0
        )

    # Apply diagonal covariance constraints where required.
    if kind in {"diag", "tied_diag"}:

        # Retain only the diagonal elements of each covariance matrix.
        for c in range(C):
            Sigma[c] = np.diag(np.diag(Sigma[c]))

    # Apply regularized discriminant analysis to every class.
    # Sigma <- lam * diag(Sigma) + (1 - lam) * Sigma
    for c in range(C):

        # Extract the diagonal component of the current covariance.
        diagonal_cov = np.diag(np.diag(Sigma[c]))

        # Blend the diagonal and full covariance estimates.
        Sigma[c] = (
            lam * diagonal_cov
            + (1.0 - lam) * Sigma[c]
        )

    # Return a new dictionary without modifying the input parameters.
    return {
        "pi": pi,
        "mu": mu,
        "Sigma": Sigma
    }

# Step 7 - lda_softmax_params
def lda_softmax_params(params):
    """
    Convert a tied-covariance GDA model into multinomial
    logistic regression weights and biases.

    Parameters
    ----------
    params : dict
        Fitted GDA parameters containing:
        - 'pi': Class prior probabilities, shape (C,).
        - 'mu': Class mean vectors, shape (C, D).
        - 'Sigma': Class covariance matrices, shape (C, D, D).

    Returns
    -------
    B : ndarray, shape (C, D)
        Weight matrix for the equivalent softmax classifier.

    gamma : ndarray, shape (C,)
        Bias vector for the equivalent softmax classifier.

    Raises
    ------
    ValueError
        If the class covariance matrices are not tied.
    """

    # Extract the class priors, means, and covariance matrices.
    pi = np.asarray(params["pi"], dtype=float)
    mu = np.asarray(params["mu"], dtype=float)
    Sigma = np.asarray(params["Sigma"], dtype=float)

    # Verify that all classes share the same covariance matrix.
    # Tied covariance is required for the quadratic terms to cancel.
    if not np.allclose(Sigma, Sigma[0]):
        raise ValueError("Class covariance matrices must be tied.")

    # Obtain the shared covariance matrix.
    shared_Sigma = Sigma[0]

    # Calculate the class-specific weight vectors.
    # Solve Sigma @ B.T = mu.T instead of explicitly computing
    # the inverse of the covariance matrix.
    B = np.linalg.solve(shared_Sigma, mu.T).T

    # Calculate the class-specific bias terms.
    # gamma_c = log(pi_c) - 0.5 * mu_c.T @ Sigma^(-1) @ mu_c
    gamma = (
        np.log(pi)
        - 0.5 * np.sum(mu * B, axis=1)
    )

    return B, gamma

# Step 8 - lda_binary_threshold
def lda_binary_threshold(params):
    """
    Convert a binary tied-covariance GDA model into a logistic
    regression representation with a prior-adjusted threshold.

    Parameters
    ----------
    params : dict
        Fitted GDA parameters containing:
        - 'pi': Class prior probabilities, shape (2,).
        - 'mu': Class mean vectors, shape (2, D).
        - 'Sigma': Class covariance matrices, shape (2, D, D).

    Returns
    -------
    w : ndarray, shape (D,)
        Direction vector of the binary LDA decision boundary.

    x0 : ndarray, shape (D,)
        Prior-adjusted threshold vector.

    Raises
    ------
    ValueError
        If the model does not contain exactly two classes
        or if the class covariance matrices are not tied.
    """

    # Extract the class priors, means, and covariance matrices.
    pi = np.asarray(params["pi"], dtype=float)
    mu = np.asarray(params["mu"], dtype=float)
    Sigma = np.asarray(params["Sigma"], dtype=float)

    # Verify that the model contains exactly two classes.
    if len(pi) != 2 or mu.shape[0] != 2 or Sigma.shape[0] != 2:
        raise ValueError("LDA binary threshold requires exactly two classes.")

    # Verify that both classes share the same covariance matrix.
    if not np.allclose(Sigma[0], Sigma[1]):
        raise ValueError("Class covariance matrices must be tied.")

    # Extract the shared covariance matrix.
    shared_Sigma = Sigma[0]

    # Calculate the difference between the two class mean vectors.
    delta_mu = mu[1] - mu[0]

    # Calculate the direction vector:
    # w = Sigma^(-1) @ (mu_1 - mu_0)
    # Use a linear system solver instead of explicitly computing the inverse.
    w = np.linalg.solve(shared_Sigma, delta_mu)

    # Calculate the squared Mahalanobis distance between class means.
    mahalanobis_sq = delta_mu @ w

    # Calculate the midpoint between the two class mean vectors.
    midpoint = 0.5 * (mu[0] + mu[1])

    # Calculate the logarithm of the class prior ratio.
    log_prior_ratio = np.log(pi[1] / pi[0])

    # Apply the prior-adjusted threshold:
    # x0 = midpoint - delta_mu * log(pi_1 / pi_0) / mahalanobis_sq
    x0 = midpoint - (
        delta_mu * log_prior_ratio / mahalanobis_sq
    )

    return w, x0

# Step 9 - ncm_log_proba
def ncm_log_proba(X, means, W):
    """
    Compute normalized log-posterior probabilities using a
    Nearest-Class-Mean classifier with a metric matrix W.

    Parameters
    ----------
    X : ndarray, shape (N, D)
        Input observations.
    means : ndarray, shape (C, D)
        Class mean vectors.
    W : ndarray, shape (K, D)
        Metric projection matrix.

    Returns
    -------
    log_proba : ndarray, shape (N, C)
        Normalized log-posterior probabilities for each observation
        and each class.
    """

    # Calculate the difference between every observation
    # and every class mean.
    # Resulting shape: (N, C, D).
    differences = X[:, None, :] - means[None, :, :]

    # Project the differences into the metric space using W.
    # Resulting shape: (N, C, K).
    projected = differences @ W.T

    # Calculate the squared Euclidean norm of each projected difference.
    # Resulting shape: (N, C).
    squared_distances = np.sum(projected ** 2, axis=-1)

    # Convert squared distances into unnormalized log scores.
    # Smaller distances produce higher log scores.
    log_scores = -0.5 * squared_distances

    # Normalize the log scores across classes using the
    # numerically stable log-sum-exp implementation.
    return log_normalize(log_scores)

