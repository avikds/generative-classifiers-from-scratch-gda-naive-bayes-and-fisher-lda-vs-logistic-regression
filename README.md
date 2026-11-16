# Generative Classifiers from Scratch: GDA, Naive Bayes and Fisher LDA vs Logistic Regression

Build the generative classifiers of Murphy's Probabilistic Machine Learning in numpy and put them up against logistic regression on MNIST. Fit Gaussian discriminant analysis by maximum likelihood, then tie, diagonalize and regularize the covariances until 30 images per class in 50 dimensions stop being singular. Read LDA's softmax weights and its prior-shifted threshold straight off the Gaussians, and learn a nearest-class-mean metric by gradient descent. Compare Fisher's discriminant with PCA and hit its C - 1 limit. Reproduce the book's 84% naive Bayes on binarized digits and prove that its posterior is a softmax of the pixels. Then test the book's claims: naive Bayes is far more confident than it is accurate, LDA learns faster than logistic regression from a few images per class while logistic regression wins on real MNIST from a hundred per class on, and when 80% of the pixels are missing, marginalizing them out keeps 77% accuracy while mean imputation drops logistic regression to 30%.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** mvn_logpdf
- [x] **2.** fit_gda
- [x] **3.** gda_log_joint
- [x] **4.** log_normalize
- [x] **5.** sample_gda
- [x] **6.** constrain_covariances
- [x] **7.** lda_softmax_params
- [x] **8.** lda_binary_threshold
- [x] **9.** ncm_log_proba
- [x] **10.** fit_ncm_metric
- [x] **11.** pca_basis
- [x] **12.** scatter_matrices
- [x] **13.** fisher_direction
- [x] **14.** fisher_projection
- [x] **15.** fisher_criterion
- [x] **16.** quantize_pixels
- [x] **17.** fit_categorical_nb
- [x] **18.** nb_log_joint
- [x] **19.** dummy_features
- [x] **20.** nb_to_softmax
- [x] **21.** fit_softmax_regression
- [x] **22.** evaluate_posteriors
- [x] **23.** learning_curve
- [x] **24.** gda_log_joint_missing

## Results

```
== Figure 9.1-9.2: three Gaussian classes in 2-D, 100 training points each ==
  Bayes-optimal test accuracy (true parameters): 0.9345
  GDA full      test accuracy 0.9338
  GDA tied      test accuracy 0.9090
  GDA diag      test accuracy 0.9285
  GDA tied_diag test accuracy 0.9133
  LDA as softmax regression (eq 9.8): max |log posterior difference| = 1.4e-14
  two-class LDA with pi1 = 0.1: x0 = [2.068, 2.132], threshold w^T x0 = 6.791
  two-class LDA with pi1 = 0.5: x0 = [1.387, 1.469], threshold w^T x0 = 4.594
  two-class LDA with pi1 = 0.9: x0 = [0.706, 0.805], threshold w^T x0 = 2.396

== GDA on 50 PCA features of MNIST (60,000 training images) ==
  full      test accuracy 0.9636
  tied      test accuracy 0.8734
  diag      test accuracy 0.8775
  tied_diag test accuracy 0.8549
  Only 30 images per class in 50 dimensions (N_c < D, eq 9.22 RDA):
    lam = 0.0  QDA singular  LDA 0.8099
    lam = 0.1  QDA 0.8727    LDA 0.8125
    lam = 0.3  QDA 0.8837    LDA 0.8086
    lam = 0.6  QDA 0.8794    LDA 0.8002
    lam = 1.0  QDA 0.7823    LDA 0.7792

== Nearest class mean on the same features (9.2.5), means from all training images ==
  Euclidean (W = I)        test accuracy 0.8175
  Mahalanobis (W = L^-1)   test accuracy 0.8744
  learned W, 60 GD steps   test accuracy 0.8799
  metric-learning NLL on 5,000 images: 0.7224 -> 0.3593

== Fisher LDA vs PCA (9.2.6): project 100 PCA features of MNIST, then fit QDA ==
  K = 2 Fisher  J(W) = 10.94   QDA test accuracy 0.5568
  K = 2 PCA     J(W) = 2.679   QDA test accuracy 0.4607
  K = 9 Fisher  J(W) = 4.759   QDA test accuracy 0.8990
  K = 9 PCA     J(W) = 3.143e-07   QDA test accuracy 0.8775
  K = 10: K cannot exceed C - 1 for Fisher LDA.

== Naive Bayes on MNIST (9.3, Figure 9.6: the book reports 84.3% for binarized pixels) ==
  2 gray levels: test accuracy 0.8427
  4 gray levels: test accuracy 0.8472
  NB as softmax regression on raw pixels (eq 9.62): max |log posterior difference| = 9.7e-13
  theta for digit 0 (Figure 9.6), every 4th pixel, '#' > 0.5, '+' > 0.1:
    ..............
    .......+++....
    ....++#####+..
    ...+##+..+##+.
    ...##+...+#+..
    ...+##+###+...
    .....+++......

== Generative vs discriminative on binarized pixels (9.4.1) ==
  naive Bayes (60,000 images)   accuracy 0.8427  log loss 3.181  ECE 0.143  confidence > 0.999 on 86.9% of test digits
  logistic regression (20,000)  accuracy 0.9101  log loss 0.308  ECE 0.009  confidence > 0.999 on 12.7% of test digits
  correlation between NB weights and LR weights (both centred over classes): 0.564

== Learning curves: test accuracy vs training images per class (3 random draws each) ==
  sizes per class: [3, 10, 30, 100, 300, 1000]
  data drawn from an LDA model (the generative model is correct):
    LDA (RDA 0.1)       0.527+-0.017  0.792+-0.015  0.861+-0.007  0.888+-0.002  0.894+-0.001  0.896+-0.000
    logistic regression 0.638+-0.016  0.773+-0.002  0.815+-0.010  0.865+-0.002  0.884+-0.000  0.893+-0.001
    Bayes-optimal accuracy 0.897
  real MNIST, 20 PCA features (the Gaussian model is wrong):
    LDA (RDA 0.1)       0.512+-0.021  0.752+-0.017  0.814+-0.002  0.836+-0.002  0.841+-0.003  0.845+-0.000
    logistic regression 0.589+-0.062  0.745+-0.015  0.802+-0.004  0.855+-0.004  0.871+-0.002  0.877+-0.001

== Exercise 10.3: training conditional log-likelihood, 1,000 images, 20 PCA features ==
  GaussI train L(M) = -0.8409   test accuracy 0.7505
  LDA    train L(M) = -0.6229   test accuracy 0.8089
  LinLog train L(M) = -0.3107   test accuracy 0.8286

== Missing features (9.4.3): pixels deleted completely at random at test time ==
  0% missing  NB marginalized: acc 0.8427 log loss 3.181   LR mean-imputed: acc 0.9101 log loss 0.308
  50% missing  NB marginalized: acc 0.8230 log loss 1.878   LR mean-imputed: acc 0.6981 log loss 0.875
  80% missing  NB marginalized: acc 0.7740 log loss 1.192   LR mean-imputed: acc 0.3050 log loss 2.246
  90% missing  NB marginalized: acc 0.6953 log loss 1.111   LR mean-imputed: acc 0.1862 log loss 2.916
  7x7 average-pooled MNIST, 23 pixels, RDA-QDA (lam 0.1) vs LR, 2,000 test digits:
  0% missing  GDA marginalized 0.8670   GDA mean-imputed 0.8670   LR mean-imputed 0.8335
  30% missing  GDA marginalized 0.7730   GDA mean-imputed 0.6180   LR mean-imputed 0.6480
  60% missing  GDA marginalized 0.5955   GDA mean-imputed 0.3650   LR mean-imputed 0.4055
  80% missing  GDA marginalized 0.3835   GDA mean-imputed 0.2175   LR mean-imputed 0.2330

Done in 27 s.
```
