// Package classic implements the classical (pre-deep-learning) machine-learning
// methods: ordinary least squares, softmax/logistic regression, k-means
// clustering, and principal component analysis.
//
// # For AI practitioners
//
// These are the workhorse baselines every practitioner still reaches for:
//
//   - [LinearRegression] — ordinary least squares via the normal equations with a
//     Cholesky solve, exposing Coef and Intercept.
//   - [SoftmaxRegression] — multinomial logistic regression (gradient descent with
//     an Adam-polished solve), returning calibrated class probabilities.
//   - [KMeans] — Lloyd's algorithm from explicit initial centers, returning final
//     centers and hard labels.
//   - [PCA] — principal components via a symmetric-eigen (Jacobi) decomposition,
//     exposing Components, ExplainedVariance (eigenvalues, descending) and Mean.
//
// Each is validated to reference tolerance against scikit-learn (§V1): OLS
// coefficients to ~1e-8, softmax probabilities to ~1e-5, k-means labels exactly,
// and PCA components up to the inherent sign ambiguity (|cosine| ≈ 1). They are
// useful in their own right and as sanity baselines a neural model must beat.
//
// # For everyone else
//
// Before neural networks, most "machine learning" was a handful of well-understood
// statistical methods, and they are still the right tool for many jobs. This
// package has four of the classics: fitting a straight line to data (linear
// regression), sorting items into categories with probabilities (softmax
// regression), grouping similar points into clusters (k-means), and finding the
// few directions that capture most of the variation in a dataset (PCA). They are
// fast, need no training loop or GPU, and are easy to check — which is why they
// make good baselines.
//
// The runnable Example functions below fit a line, cluster points, and measure how
// much variance a principal component captures.
//
// Further reading: Bishop, "Pattern Recognition and Machine Learning" (Springer 2006), and Hastie, Tibshirani & Friedman, "The Elements of Statistical Learning" (2nd ed.) — the standard references for the classical methods here.
package classic
