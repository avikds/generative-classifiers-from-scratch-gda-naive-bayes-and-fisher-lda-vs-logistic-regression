"""
Generative Classifiers from Scratch: GDA, Naive Bayes and Fisher LDA vs Logistic Regression scaffold.

Run this with: python scaffold.py
Uses functions defined in model.py.
"""

from model import *  # noqa: F401, F403 (pulls in your solution functions)

"""Generative Classifiers from Scratch (PML Book 1, chapter 9).
GDA with full / tied / diagonal / RDA covariances, LDA as softmax, nearest class mean metric learning,
Fisher LDA vs PCA, naive Bayes on binarized MNIST, and the generative-vs-discriminative comparison of section 9.4."""
import os
import tempfile
import time
import urllib.request
import numpy as np

from solution import (mvn_logpdf, fit_gda, gda_log_joint, log_normalize, sample_gda, constrain_covariances,
                      lda_softmax_params, lda_binary_threshold, ncm_log_proba, fit_ncm_metric, pca_basis,
                      scatter_matrices, fisher_direction, fisher_projection, fisher_criterion, quantize_pixels,
                      fit_categorical_nb, nb_log_joint, dummy_features, nb_to_softmax, fit_softmax_regression,
                      evaluate_posteriors, learning_curve, gda_log_joint_missing)


def load_mnist():
    path = os.path.join(tempfile.gettempdir(), "mnist.npz")
    if not os.path.exists(path):
        urllib.request.urlretrieve("https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz", path)
    with np.load(path) as z:
        return z["x_train"], z["y_train"].astype(np.int64), z["x_test"], z["y_test"].astype(np.int64)


def acc(log_scores, y):
    return float(np.mean(np.argmax(log_scores, axis=1) == y))


def fmt(curve):
    mean, std = curve
    return "  ".join(f"{m:.3f}+-{s:.3f}" for m, s in zip(mean, std))


def main():
    t0 = time.time()
    # --- Part 1-2 on toy data: three 2-D classes like Figure 9.1 ---
    truth = {"pi": np.ones(3) / 3, "mu": np.array([[0., 0.], [3., 3.], [4., -1.]]),
             "Sigma": np.array([[[2., 1.5], [1.5, 2.]], [[1., 0.], [0., 2.5]], [[0.4, 0.], [0., 0.4]]])}
    Xtr, ytr = sample_gda(truth, 100, np.random.default_rng(0))
    Xte, yte = sample_gda(truth, 2000, np.random.default_rng(1))
    gda = fit_gda(Xtr, ytr, 3)
    print("== Figure 9.1-9.2: three Gaussian classes in 2-D, 100 training points each ==")
    print(f"  Bayes-optimal test accuracy (true parameters): {acc(gda_log_joint(truth, Xte), yte):.4f}")
    for kind in ("full", "tied", "diag", "tied_diag"):
        q = constrain_covariances(gda, kind)
        print(f"  GDA {kind:9s} test accuracy {acc(gda_log_joint(q, Xte), yte):.4f}")
    lda = constrain_covariances(gda, "tied")
    B, gamma = lda_softmax_params(lda)
    gap = np.abs(log_normalize(Xte @ B.T + gamma) - log_normalize(gda_log_joint(lda, Xte))).max()
    print(f"  LDA as softmax regression (eq 9.8): max |log posterior difference| = {gap:.1e}")
    two = {"pi": np.array([0.5, 0.5]), "mu": lda["mu"][:2], "Sigma": lda["Sigma"][:2]}
    for p1 in (0.1, 0.5, 0.9):
        two["pi"] = np.array([1 - p1, p1])
        w, x0 = lda_binary_threshold(two)
        print(f"  two-class LDA with pi1 = {p1}: x0 = {np.round(x0, 3).tolist()}, threshold w^T x0 = {w @ x0:.3f}")

    # --- MNIST ---
    x_tr, y_tr, x_te, y_te = load_mnist()
    flat_tr, flat_te = x_tr.reshape(60000, -1) / 255.0, x_te.reshape(10000, -1) / 255.0
    mean50, V50 = pca_basis(flat_tr, 50)
    Z, Zt = (flat_tr - mean50) @ V50, (flat_te - mean50) @ V50
    print("\n== GDA on 50 PCA features of MNIST (60,000 training images) ==")
    g50 = fit_gda(Z, y_tr, 10)
    for kind in ("full", "tied", "diag", "tied_diag"):
        print(f"  {kind:9s} test accuracy {acc(gda_log_joint(constrain_covariances(g50, kind), Zt), y_te):.4f}")
    rng = np.random.default_rng(0)
    idx = np.concatenate([rng.choice(np.flatnonzero(y_tr == c), 30, replace=False) for c in range(10)])
    small = fit_gda(Z[idx], y_tr[idx], 10)
    print("  Only 30 images per class in 50 dimensions (N_c < D, eq 9.22 RDA):")
    for lam in (0.0, 0.1, 0.3, 0.6, 1.0):
        try:
            a = f"{acc(gda_log_joint(constrain_covariances(small, 'full', lam), Zt), y_te):.4f}"
        except np.linalg.LinAlgError:
            a = "singular"
        print(f"    lam = {lam:.1f}  QDA {a:9s} LDA {acc(gda_log_joint(constrain_covariances(small, 'tied', lam), Zt), y_te):.4f}")

    print("\n== Nearest class mean on the same features (9.2.5), means from all training images ==")
    sub = np.arange(5000)
    means = g50["mu"]
    W_mahal = np.linalg.inv(np.linalg.cholesky(constrain_covariances(g50, "tied")["Sigma"][0]))
    W_learn, losses = fit_ncm_metric(Z[sub], y_tr[sub], means, 50, n_iter=60, lr=0.05, seed=0)
    for name, W in (("Euclidean (W = I)", np.eye(50)), ("Mahalanobis (W = L^-1)", W_mahal), ("learned W, 60 GD steps", W_learn)):
        print(f"  {name:24s} test accuracy {acc(ncm_log_proba(Zt, means, W), y_te):.4f}")
    print(f"  metric-learning NLL on 5,000 images: {losses[0]:.4f} -> {losses[-1]:.4f}")

    print("\n== Fisher LDA vs PCA (9.2.6): project 100 PCA features of MNIST, then fit QDA ==")
    mean100, V100 = pca_basis(flat_tr, 100)
    Z100, Z100t = (flat_tr - mean100) @ V100, (flat_te - mean100) @ V100
    for K in (2, 9):
        Wf = fisher_projection(Z100, y_tr, 10, K)
        Wp = np.eye(100)[:, :K]
        for name, W in (("Fisher", Wf), ("PCA", Wp)):
            q = fit_gda(Z100 @ W, y_tr, 10)
            print(f"  K = {K} {name:6s}  J(W) = {fisher_criterion(W, Z100, y_tr, 10):.4g}   QDA test accuracy {acc(gda_log_joint(q, Z100t @ W), y_te):.4f}")
    try:
        fisher_projection(Z100, y_tr, 10, 10)
    except ValueError as e:
        print("  K = 10:", e)

    print("\n== Naive Bayes on MNIST (9.3, Figure 9.6: the book reports 84.3% for binarized pixels) ==")
    nbs = {}
    for K in (2, 4):
        Xk, Xk_te = quantize_pixels(x_tr, K), quantize_pixels(x_te, K)
        nb = fit_categorical_nb(Xk, y_tr, 10, K)
        nbs[K] = (nb, Xk, Xk_te)
        print(f"  {K} gray levels: test accuracy {acc(nb_log_joint(nb, Xk_te), y_te):.4f}")
    nb, Xb, Xb_te = nbs[2]
    B, gamma = nb_to_softmax(nb)
    gap = np.abs(log_normalize(dummy_features(Xb_te, 2) @ B.T + gamma) - log_normalize(nb_log_joint(nb, Xb_te))).max()
    print(f"  NB as softmax regression on raw pixels (eq 9.62): max |log posterior difference| = {gap:.1e}")
    theta = np.exp(nb["log_theta"][:, :, 1]).reshape(10, 28, 28)
    print("  theta for digit 0 (Figure 9.6), every 4th pixel, '#' > 0.5, '+' > 0.1:")
    for row in theta[0, ::4, ::2]:
        print("    " + "".join("#" if v > 0.5 else "+" if v > 0.1 else "." for v in row))

    print("\n== Generative vs discriminative on binarized pixels (9.4.1) ==")
    lr = fit_softmax_regression(Xb[:20000].astype(float), y_tr[:20000], 10, lam=1e-4, n_iter=200, lr=0.5)
    lp_lr = log_normalize(Xb_te @ lr["W"].T + lr["b"])
    lp_nb = log_normalize(nb_log_joint(nb, Xb_te))
    for name, lp in (("naive Bayes (60,000 images)", lp_nb), ("logistic regression (20,000)", lp_lr)):
        r = evaluate_posteriors(lp, y_te)
        print(f"  {name:29s} accuracy {r['accuracy']:.4f}  log loss {r['nll']:.3f}  ECE {r['ece']:.3f}  "
              f"confidence > 0.999 on {np.mean(np.exp(lp.max(1)) > 0.999):.1%} of test digits")
    corr = np.corrcoef((B - B.mean(0)).ravel(), (lr["W"] - lr["W"].mean(0)).ravel())[0, 1]
    print(f"  correlation between NB weights and LR weights (both centred over classes): {corr:.3f}")

    print("\n== Learning curves: test accuracy vs training images per class (3 random draws each) ==")
    sizes = [3, 10, 30, 100, 300, 1000]
    Z20, Z20t = Z[:, :20], Zt[:, :20]
    model20 = constrain_covariances(fit_gda(Z20, y_tr, 10), "tied")
    Xs, ys = sample_gda(model20, 1500, np.random.default_rng(1))
    Xst, yst = sample_gda(model20, 1000, np.random.default_rng(2))

    def lda_fp(A, a, T):
        return gda_log_joint(constrain_covariances(fit_gda(A, a, 10), "tied", 0.1), T)

    def lr_fp(A, a, T):
        m = fit_softmax_regression(A, a, 10, lam=1e-4, n_iter=300, lr=0.1)
        return T @ m["W"].T + m["b"]

    print("  sizes per class:", sizes)
    print("  data drawn from an LDA model (the generative model is correct):")
    print("    LDA (RDA 0.1)      ", fmt(learning_curve(lda_fp, Xs, ys, Xst, yst, sizes, 3, seed=0)))
    print("    logistic regression", fmt(learning_curve(lr_fp, Xs, ys, Xst, yst, sizes, 3, seed=0)))
    print(f"    Bayes-optimal accuracy {acc(gda_log_joint(model20, Xst), yst):.3f}")
    print("  real MNIST, 20 PCA features (the Gaussian model is wrong):")
    print("    LDA (RDA 0.1)      ", fmt(learning_curve(lda_fp, Z20, y_tr, Z20t, y_te, sizes, 3, seed=0)))
    print("    logistic regression", fmt(learning_curve(lr_fp, Z20, y_tr, Z20t, y_te, sizes, 3, seed=0)))

    print("\n== Exercise 10.3: training conditional log-likelihood, 1,000 images, 20 PCA features ==")
    n = np.arange(1000)
    gauss_i = fit_gda(Z20[n], y_tr[n], 10)
    gauss_i["Sigma"], gauss_i["pi"] = np.array([np.eye(20)] * 10), np.ones(10) / 10
    lda20 = constrain_covariances(fit_gda(Z20[n], y_tr[n], 10), "tied")
    lin = fit_softmax_regression(Z20[n], y_tr[n], 10, lam=0.0, n_iter=2000, lr=0.1)
    for name, f in (("GaussI", lambda T: gda_log_joint(gauss_i, T)), ("LDA", lambda T: gda_log_joint(lda20, T)),
                    ("LinLog", lambda T: T @ lin["W"].T + lin["b"])):
        tr = evaluate_posteriors(log_normalize(f(Z20[n])), y_tr[n])
        te = evaluate_posteriors(log_normalize(f(Z20t)), y_te)
        print(f"  {name:6s} train L(M) = {-tr['nll']:.4f}   test accuracy {te['accuracy']:.4f}")

    print("\n== Missing features (9.4.3): pixels deleted completely at random at test time ==")
    pix_mean = Xb.mean(axis=0)
    rng = np.random.default_rng(3)
    for rate in (0.0, 0.5, 0.8, 0.9):
        M = rng.random(Xb_te.shape) < rate
        a_nb = evaluate_posteriors(log_normalize(nb_log_joint(nb, np.where(M, -1, Xb_te))), y_te)
        a_lr = evaluate_posteriors(log_normalize(np.where(M, pix_mean, Xb_te) @ lr["W"].T + lr["b"]), y_te)
        print(f"  {rate:.0%} missing  NB marginalized: acc {a_nb['accuracy']:.4f} log loss {a_nb['nll']:.3f}   "
              f"LR mean-imputed: acc {a_lr['accuracy']:.4f} log loss {a_lr['nll']:.3f}")
    P = x_tr.reshape(60000, 7, 4, 7, 4).mean(axis=(2, 4)).reshape(60000, 49) / 255.0
    Pt = x_te.reshape(10000, 7, 4, 7, 4).mean(axis=(2, 4)).reshape(10000, 49) / 255.0
    keep = np.min([P[y_tr == c].std(axis=0) for c in range(10)], axis=0) > 0.01
    P, Pt = P[:, keep], Pt[:, keep][:2000]
    qp = constrain_covariances(fit_gda(P, y_tr, 10), "full", 0.1)
    lrp = fit_softmax_regression(P, y_tr, 10, lam=1e-4, n_iter=300, lr=0.5)
    pm = P.mean(axis=0)
    print(f"  7x7 average-pooled MNIST, {int(keep.sum())} pixels, RDA-QDA (lam 0.1) vs LR, 2,000 test digits:")
    for rate in (0.0, 0.3, 0.6, 0.8):
        M = rng.random(Pt.shape) < rate
        a_m = acc(gda_log_joint_missing(qp, np.where(M, np.nan, Pt)), y_te[:2000])
        a_i = acc(gda_log_joint(qp, np.where(M, pm, Pt)), y_te[:2000])
        a_l = acc(np.where(M, pm, Pt) @ lrp["W"].T + lrp["b"], y_te[:2000])
        print(f"  {rate:.0%} missing  GDA marginalized {a_m:.4f}   GDA mean-imputed {a_i:.4f}   LR mean-imputed {a_l:.4f}")
    print(f"\nDone in {time.time() - t0:.0f} s.")


if __name__ == "__main__":
    main()

