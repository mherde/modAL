import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.utils import check_random_state, check_array
 

class PWC(BaseEstimator, ClassifierMixin):
    """PWC
    The Parzen window classifier (PWC) is a simple and probabilistic classifier. This classifier is based on a
    non-parametric density estimation obtained by applying a kernel function.

    Parameters
    ----------
    n_classes: int,
        This parameter indicates the number of available classes.
    metric: str,
        The metric must a be a valid kernel defined by the function sklearn.metrics.pairwise.pairwise_kernels.
    n_neighbors: int,
        Number of nearest neighbours. Default is None, which means all available samples are considered.
    kwargs: str,
        Any further parameters are passed directly to the kernel function.

    Attributes
    ----------
    n_classes: int,
        This parameters indicates the number of available classes.
    metric: str,
        The metric must a be a valid kernel defined by the function sklearn.metrics.pairwise.pairwise_kernels.
    n_neighbors: int,
        Number of nearest neighbours. Default is None, which means all available samples are considered.
    kwargs: str,
        Any further parameters are passed directly to the kernel function.
    X: array-like, shape (n_samples, n_features)
        The sample matrix X is the feature matrix representing the samples.
    y: array-like, shape (n_samples) or (n_samples, n_outputs)
        It contains the class labels of the training samples.
        The number of class labels may be variable for the samples.
    Z: array-like, shape (n_samples, n_classes)
        The class labels are represented by counting vectors. An entry Z[i,j] indicates how many class labels of class j
        were provided for training sample i.

    References
    ----------
    O. Chapelle, "Active Learning for Parzen Window Classifier",
    Proceedings of the Tenth International Workshop Artificial Intelligence and Statistics, 2005.
    """
    
    # see https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.pairwise_kernels.html for
    # more information
    KERNEL_PARAMS = {
        "additive_chi2": (),
        "chi2": frozenset(["gamma"]),
        "cosine": (),
        "linear": (),
        "poly": frozenset(["gamma", "degree", "coef0"]),
        "polynomial": frozenset(["gamma", "degree", "coef0"]),
        "rbf": frozenset(["gamma"]),
        "laplacian": frozenset(["gamma"]),
        "sigmoid": frozenset(["gamma", "coef0"]),
        "precomputed": frozenset([])
    }

    def __init__(self, n_classes, metric='rbf', n_neighbors=None, gamma=None, random_state=42, **kwargs):
        self.n_classes = int(n_classes)
        if self.n_classes <= 0:
            raise ValueError("The parameter 'n_classes' must be a positive integer.")
        self.metric = str(metric)
        if self.metric not in self.KERNEL_PARAMS.keys():
            raise ValueError("The parameter 'metric' must be a in {}".format(self.KERNEL_PARAMS.keys()))
        self.n_neighbors = int(n_neighbors) if n_neighbors is not None else n_neighbors
        if self.n_neighbors is not None and self.n_neighbors <= 0:
            raise ValueError("The parameter 'n_neighbors' must be a positive integer.")
        self.random_state = check_random_state(random_state)
        self.kwargs = kwargs
        self.X = None
        self.y = None
        self.Z = None
        self.gamma = gamma

    def fit(self, X, y):
        """
        Fit the model using X as training data and y as class labels.

        Parameters
        ----------
        X: matrix-like, shape (n_samples, n_features)
            The sample matrix X is the feature matrix representing the samples.
        y: array-like, shape (n_samples) or (n_samples, n_outputs)
            It contains the class labels of the training samples.
            The number of class labels may be variable for the samples, where missing labels are
            represented by np.nan.

        Returns
        -------
        self: PWC,
            The PWC is fitted on the training data.
        """
        if np.size(X) > 0:
            self.X = check_array(X)
            self.y = check_array(y, ensure_2d=False, force_all_finite=False).astype(int)

            # convert labels to count vectors
            self.Z = np.zeros((np.size(X, 0), self.n_classes))
            for i in range(np.size(self.Z, 0)):
                self.Z[i, self.y[i]] += 1

        return self

    def predict_proba(self, X, **kwargs):
        """
        Return probability estimates for the test data X.

        Parameters
        ----------
        X:  array-like, shape (n_samples, n_features) or shape (n_samples, m_samples) if metric == 'precomputed'
            Test samples.
        C: array-like, shape (n_classes, n_classes)
            Classification cost matrix.

        Returns
        -------
        P:  array-like, shape (t_samples, n_classes)
            The class probabilities of the input samples. Classes are ordered by lexicographic order.
        """
        # if normalize is false, the probabilities are frequency estimates
        normalize = kwargs.pop('normalize', True)

        # no training data -> random prediction
        if self.X is None or np.size(self.X, 0) == 0:
            if normalize:
                return np.full((np.size(X, 0), self.n_classes), 1. / self.n_classes)
            else:
                return np.zeros((np.size(X, 0), self.n_classes))

        # calculating metric matrix
        if self.metric == 'linear':
            K = pairwise_kernels(X, self.X, metric=self.metric, **self.kwargs)
        else:
            K = pairwise_kernels(X, self.X, metric=self.metric, gamma=self.gamma, **self.kwargs)

        if self.n_neighbors is None:
            # calculating labeling frequency estimates
            P = K @ self.Z
        else:
            if np.size(self.X, 0) < self.n_neighbors:
                n_neighbors = np.size(self.X, 0)
            else:
                n_neighbors = self.n_neighbors
            indices = np.argpartition(K, -n_neighbors, axis=1)[:, -n_neighbors:]
            P = np.empty((np.size(X, 0), self.n_classes))
            for i in range(np.size(X, 0)):
                P[i, :] = K[i, indices[i]] @ self.Z[indices[i], :]

        if normalize:
            # normalizing probabilities of each sample
            normalizer = np.sum(P, axis=1)
            P[normalizer > 0] /= normalizer[normalizer > 0, np.newaxis]
            P[normalizer == 0, :] = [1 / self.n_classes] * self.n_classes
            # normalizer[normalizer == 0.0] = 1.0
            # for y_idx in range(self.n_classes):
            #     P[:, y_idx] /= normalizer

        return P

    def predict(self, X, **kwargs):
        """
        Return class label predictions for the test data X.

        Parameters
        ----------
        X:  array-like, shape (n_samples, n_features) or shape (n_samples, m_samples) if metric == 'precomputed'
            Test samples.

        Returns
        -------
        y:  array-like, shape = [n_samples]
            Predicted class labels class.
        """
        C = kwargs.pop('C', None)

        if C is None:
            C = np.ones((self.n_classes, self.n_classes))
            np.fill_diagonal(C, 0)

        P = self.predict_proba(X, normalize=True)
        return self._rand_arg_min(np.dot(P, C), axis=1)

    def reset(self):
        """
        Reset fitted parameters.
        """
        self.X = None
        self.y = None
        self.Z = None
        self.random_state = self.random_state

    def _rand_arg_min(self, arr, axis=1):
        """
        Returns index of minimal element per given axis. In case of a tie, the index is chosen randomly.
        
        Parameters
        ----------
        arr: array-like
        Array whose minimal elements' indices are determined.
        axis: int
        Indices of minimal elements are determined along this axis.

        Returns
        -------
        min_indices: array-like
        Indices of minimal elements.
        """
        arr_min = arr.min(axis, keepdims=True)
        tmp = self.random_state.uniform(low=1, high=2, size=arr.shape) * (arr == arr_min)
        return tmp.argmax(axis)
