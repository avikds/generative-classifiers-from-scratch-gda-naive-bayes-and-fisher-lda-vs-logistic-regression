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

# Step 10 - fit_ncm_metric
def fit_ncm_metric(X, y, means, K, n_iter=100, lr=0.1, seed=0):
    """
    Learn the metric matrix W for a Nearest-Class-Mean classifier
    using gradient descent on the negative log-likelihood.

    Parameters
    ----------
    X : ndarray, shape (N, D)
        Training observations.
    y : ndarray, shape (N,)
        Integer class labels ranging from 0 to C-1.
    means : ndarray, shape (C, D)
        Fixed class mean vectors.
    K : int
        Dimensionality of the learned projection space.
    n_iter : int, default=100
        Number of gradient descent iterations.
    lr : float, default=0.1
        Learning rate.
    seed : int, default=0
        Random seed for reproducible initialization.

    Returns
    -------
    W : ndarray, shape (K, D)
        Learned metric projection matrix.
    losses : ndarray, shape (n_iter,)
        Negative log-likelihood recorded before each update.
    """

    # Determine the number of observations and input features.
    N, D = X.shape

    # Determine the number of classes.
    C = means.shape[0]

    # Initialize the random number generator for reproducibility.
    rng = np.random.default_rng(seed)

    # Initialize W as a rectangular identity matrix with small random noise.
    W = np.eye(K, D) + 0.01 * rng.standard_normal((K, D))

    # Store the negative log-likelihood at every iteration.
    losses = np.zeros(n_iter, dtype=float)

    # Create the one-hot encoded representation of class labels.
    onehot = np.zeros((N, C), dtype=float)
    onehot[np.arange(N), y] = 1.0

    # Perform gradient descent for the specified number of iterations.
    for i in range(n_iter):

        # Compute normalized log-posterior probabilities.
        logp = ncm_log_proba(X, means, W)

        # Record the negative log-likelihood before updating W.
        losses[i] = -np.mean(logp[np.arange(N), y])

        # Calculate the prediction error for each observation and class.
        # g has shape (N, C).
        # Dividing by N gives the gradient of the mean loss.
        g = (np.exp(logp) - onehot) / N

        # Initialize the accumulated covariance-like gradient matrix.
        G = np.zeros((D, D), dtype=float)

        # Accumulate the weighted outer products for every class.
        for c in range(C):

            # Compute differences between observations and class c's mean.
            # Shape: (N, D).
            R_c = X - means[c]

            # Accumulate:
            # G += sum_n g[n, c] * R_c[n] @ R_c[n].T
            G += R_c.T @ (R_c * g[:, c:c+1])

        # Apply the gradient descent update.
        # Since dL/dW = -W @ G, subtracting the gradient
        # is equivalent to adding the term below.
        W = W + lr * (W @ G)

    # Return the learned metric and the recorded loss history.
    return W, losses

# Step 11 - pca_basis
def pca_basis(X, K):
    """
    Compute the top-K principal component directions of X.

    Parameters
    ----------
    X : ndarray, shape (N, D)
        Input dataset containing N observations and D features.
    K : int
        Number of principal components to retain.

    Returns
    -------
    mean : ndarray, shape (D,)
        Mean of each feature.
    V : ndarray, shape (D, K)
        Eigenvectors corresponding to the K largest eigenvalues,
        ordered from largest to smallest, with orthonormal columns.
    """

    # Convert the input dataset to a floating-point NumPy array.
    X = np.asarray(X, dtype=float)

    # Compute the column-wise mean of the dataset.
    mean = np.mean(X, axis=0)

    # Center the observations by subtracting the feature means.
    X_centered = X - mean

    # Compute the covariance matrix using the maximum likelihood
    # normalization factor 1/N.
    covariance = (X_centered.T @ X_centered) / X.shape[0]

    # Compute the eigenvalues and eigenvectors of the symmetric
    # covariance matrix.
    # np.linalg.eigh returns eigenvalues in ascending order.
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)

    # Select the K eigenvectors corresponding to the largest
    # eigenvalues, arranging them in descending order.
    V = eigenvectors[:, ::-1][:, :K]

    # Fix the sign ambiguity of eigenvectors.
    # Ensure that the largest-magnitude entry in every column is positive.
    for j in range(K):

        # Find the index of the largest absolute entry in the current column.
        max_idx = np.argmax(np.abs(V[:, j]))

        # Flip the sign if the selected entry is negative.
        if V[max_idx, j] < 0:
            V[:, j] *= -1.0

    return mean, V

# Step 12 - scatter_matrices
def scatter_matrices(X, y, C):
    """
    Compute the within-class and between-class scatter matrices.

    Parameters
    ----------
    X : ndarray, shape (N, D)
        Input dataset containing N observations and D features.
    y : ndarray, shape (N,)
        Integer class labels ranging from 0 to C-1.
    C : int
        Number of classes.

    Returns
    -------
    S_W : ndarray, shape (D, D)
        Within-class scatter matrix.
    S_B : ndarray, shape (D, D)
        Between-class scatter matrix.
    """

    # Convert inputs to NumPy arrays.
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)

    # Determine the number of observations and features.
    N, D = X.shape

    # Calculate the overall mean of the entire dataset.
    overall_mean = np.mean(X, axis=0)

    # Initialize the within-class and between-class scatter matrices.
    S_W = np.zeros((D, D), dtype=float)
    S_B = np.zeros((D, D), dtype=float)

    # Calculate scatter contributions for each class.
    for c in range(C):

        # Select observations belonging to the current class.
        X_c = X[y == c]

        # Count the observations in the current class.
        N_c = X_c.shape[0]

        # Skip empty classes because they contribute zero scatter.
        if N_c == 0:
            continue

        # Calculate the mean vector of the current class.
        mu_c = np.mean(X_c, axis=0)

        # Center observations around their respective class mean.
        centered_X = X_c - mu_c

        # Accumulate the within-class scatter matrix.
        # S_W = sum_c sum_n (x_n - mu_c)(x_n - mu_c).T
        S_W += centered_X.T @ centered_X

        # Calculate the difference between the class mean
        # and the overall dataset mean.
        mean_diff = mu_c - overall_mean

        # Accumulate the between-class scatter matrix.
        # S_B = sum_c N_c * (mu_c - mu)(mu_c - mu).T
        S_B += N_c * np.outer(mean_diff, mean_diff)

    # Return both scatter matrices.
    return S_W, S_B

# Step 13 - fisher_direction
def fisher_direction(X, y):
    """
    Compute Fisher's optimal one-dimensional projection
    direction for a two-class classification problem.

    Parameters
    ----------
    X : ndarray, shape (N, D)
        Input dataset containing N observations and D features.
    y : ndarray, shape (N,)
        Binary class labels containing 0 and 1.

    Returns
    -------
    w : ndarray, shape (D,)
        Unit-length Fisher discriminant direction.
    """

    # Convert inputs to NumPy arrays.
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)

    # Compute the within-class scatter matrix.
    # The between-class scatter matrix is not needed here.
    S_W, _ = scatter_matrices(X, y, 2)

    # Calculate the mean vector of each class.
    mu_0 = np.mean(X[y == 0], axis=0)
    mu_1 = np.mean(X[y == 1], axis=0)

    # Calculate the difference between the two class means.
    mean_diff = mu_1 - mu_0

    # Solve the linear system instead of explicitly
    # calculating the inverse of the within-class scatter matrix.
    w = np.linalg.solve(S_W, mean_diff)

    # Normalize the discriminant direction to unit length.
    w = w / np.linalg.norm(w)

    return w

# Step 14 - fisher_projection
def fisher_projection(X, y, C, K):
    """
    Compute the multi-class Fisher LDA projection matrix.

    Parameters
    ----------
    X : ndarray, shape (N, D)
        Input dataset containing N observations and D features.
    y : ndarray, shape (N,)
        Integer class labels ranging from 0 to C-1.
    C : int
        Number of classes.
    K : int
        Number of discriminant dimensions to retain.

    Returns
    -------
    W : ndarray, shape (D, K)
        Fisher LDA projection matrix with orthonormalized
        within-class whitening directions.

    Raises
    ------
    ValueError
        If K exceeds C - 1.
    """

    # Fisher LDA can produce at most C - 1 discriminant directions.
    if K > C - 1:
        raise ValueError("K cannot exceed C - 1 for Fisher LDA.")

    # Compute the within-class and between-class scatter matrices.
    S_W, S_B = scatter_matrices(X, y, C)

    # Eigendecompose the within-class scatter matrix.
    # S_W = Q @ diag(e) @ Q.T
    e, Q = np.linalg.eigh(S_W)

    # Construct the inverse square root of S_W.
    # S_W^(-1/2) = Q @ diag(e^(-1/2)) @ Q.T
    S_W_inv_sqrt = (
        Q @ np.diag(1.0 / np.sqrt(e)) @ Q.T
    )

    # Whiten the between-class scatter matrix.
    # M = S_W^(-1/2) @ S_B @ S_W^(-1/2)
    M = S_W_inv_sqrt @ S_B @ S_W_inv_sqrt

    # Symmetrize M to eliminate small numerical asymmetries.
    M = 0.5 * (M + M.T)

    # Eigendecompose the whitened between-class scatter matrix.
    # np.linalg.eigh returns eigenvalues in ascending order.
    eigenvalues, U = np.linalg.eigh(M)

    # Select the K eigenvectors corresponding to the largest
    # eigenvalues, arranging them in descending order.
    U = U[:, ::-1][:, :K]

    # Transform the selected eigenvectors back into the
    # original feature space.
    W = S_W_inv_sqrt @ U

    # Fix the sign ambiguity of each projection direction.
    # Ensure the largest-magnitude entry in every column is positive.
    for j in range(K):

        # Find the index of the largest absolute entry.
        max_idx = np.argmax(np.abs(W[:, j]))

        # Flip the column if its largest-magnitude entry is negative.
        if W[max_idx, j] < 0:
            W[:, j] *= -1.0

    return W

# Step 15 - fisher_criterion
def fisher_criterion(W, X, y, C):
    """
    Evaluate a linear projection using Fisher's determinant criterion.

    Parameters
    ----------
    W : ndarray, shape (D, K) or (D,)
        Projection matrix. A one-dimensional vector is treated
        as a single projection direction of shape (D, 1).
    X : ndarray, shape (N, D)
        Input dataset containing N observations and D features.
    y : ndarray, shape (N,)
        Integer class labels ranging from 0 to C-1.
    C : int
        Number of classes.

    Returns
    -------
    criterion : float
        Fisher's determinant criterion, calculated as the ratio
        of the determinant of projected between-class scatter
        to the determinant of projected within-class scatter.
    """

    # Convert the projection matrix into a floating-point NumPy array.
    W = np.asarray(W, dtype=float)

    # Treat a one-dimensional projection vector as a single-column matrix.
    if W.ndim == 1:
        W = W.reshape(-1, 1)

    # Compute the within-class and between-class scatter matrices.
    S_W, S_B = scatter_matrices(X, y, C)

    # Project the between-class scatter matrix into the reduced space.
    projected_S_B = W.T @ S_B @ W

    # Project the within-class scatter matrix into the reduced space.
    projected_S_W = W.T @ S_W @ W

    # Calculate the determinants of the projected scatter matrices.
    det_SB = np.linalg.det(projected_S_B)
    det_SW = np.linalg.det(projected_S_W)

    # Compute Fisher's determinant criterion.
    criterion = det_SB / det_SW

    # Return the result as a standard Python float.
    return float(criterion)

# Step 16 - quantize_pixels
def quantize_pixels(images, K):
    """
    Flatten 8-bit grayscale images and quantize their pixels
    into K discrete intensity levels.

    Parameters
    ----------
    images : ndarray, shape (N, H, W) or (N, D)
        Input images with integer pixel values between 0 and 255.
    K : int
        Number of discrete gray levels (K >= 2).

    Returns
    -------
    quantized : ndarray, shape (N, H*W) or (N, D)
        Flattened and quantized pixel values with dtype int64.
        Each pixel belongs to one of the levels 0, 1, ..., K-1.
    """

    # Convert the input images into a NumPy array.
    images = np.asarray(images)

    # Flatten every image into a one-dimensional feature vector.
    # The first dimension represents the number of images.
    pixels = images.reshape(images.shape[0], -1)

    # Convert pixel values to int64 before multiplication
    # to prevent integer overflow with uint8 input arrays.
    pixels = pixels.astype(np.int64)

    # Quantize each pixel into one of K discrete gray levels.
    # The resulting values range from 0 to K-1.
    quantized = (pixels * K) // 256

    return quantized

# Step 17 - fit_categorical_nb
def fit_categorical_nb(Xk, y, C, K, alpha=1.0, beta=1.0):
    """
    Fit a categorical Naive Bayes classifier using frequency
    counting with Dirichlet pseudo-count smoothing.

    Parameters
    ----------
    Xk : ndarray, shape (N, D)
        Integer-valued categorical features ranging from 0 to K-1.
        A value of -1 represents a missing feature.
    y : ndarray, shape (N,)
        Integer class labels ranging from 0 to C-1.
    C : int
        Number of classes.
    K : int
        Number of possible categorical feature values.
    alpha : float, default=1.0
        Dirichlet pseudo-count for class prior probabilities.
    beta : float, default=1.0
        Dirichlet pseudo-count for feature conditional probabilities.

    Returns
    -------
    dict
        Dictionary containing:
        - 'log_theta': Log conditional probabilities,
          shape (C, D, K).
        - 'log_pi': Log class prior probabilities,
          shape (C,).
    """

    # Convert inputs into NumPy arrays.
    Xk = np.asarray(Xk, dtype=int)
    y = np.asarray(y, dtype=int)

    # Determine the number of observations and features.
    N, D = Xk.shape

    # Create one-hot encoded class labels.
    # Shape: (N, C).
    onehot = np.eye(C, dtype=float)[y]

    # Initialize the categorical feature count tensor.
    # Shape: (C, D, K).
    counts = np.zeros((C, D, K), dtype=float)

    # Count occurrences of every categorical level.
    # Missing values (-1) are automatically excluded because
    # they do not match any of the valid levels 0, ..., K-1.
    for k in range(K):

        # Identify observations whose feature values equal k.
        feature_mask = (Xk == k).astype(float)

        # Aggregate feature counts independently for every class.
        # One matrix multiplication per categorical level.
        counts[:, :, k] = onehot.T @ feature_mask

    # Calculate the conditional probability denominators.
    # Each feature is normalized using its observed counts
    # plus K Dirichlet pseudo-counts.
    feature_totals = (
        counts.sum(axis=2, keepdims=True) + K * beta
    )

    # Apply Dirichlet smoothing to obtain conditional probabilities.
    theta = (counts + beta) / feature_totals

    # Convert conditional probabilities into log probabilities.
    log_theta = np.log(theta)

    # Count the number of observations belonging to each class.
    N_c = onehot.sum(axis=0)

    # Estimate the class prior probabilities using Dirichlet smoothing.
    pi = (N_c + alpha) / (N + C * alpha)

    # Convert class prior probabilities into log probabilities.
    log_pi = np.log(pi)

    # Return the fitted categorical Naive Bayes parameters.
    return {
        "log_theta": log_theta,
        "log_pi": log_pi
    }

