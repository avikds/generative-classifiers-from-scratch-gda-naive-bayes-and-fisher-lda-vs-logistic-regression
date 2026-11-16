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

---

Built on Deep-ML.
