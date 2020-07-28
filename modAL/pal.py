import numpy as np
import itertools
from scipy.special import factorial, gammaln
from modAL.utils.parzen_window_classifier import PWC


from typing import Tuple

import numpy as np
from scipy.stats import entropy
from sklearn.exceptions import NotFittedError
from sklearn.base import BaseEstimator

from modAL.utils.data import modALinput
from modAL.utils.selection import multi_argmax, shuffled_argmax


def cost_reduction(k_vec_list, C=None, m_max=2, prior=1.e-3):
    """Calculates the expected cost reduction for given maximal number of hypothetically acquired labels,
        observed labels and cost matrix.

    Parameters
    ----------
    k_vec_list: array-like, shape [n_classes]
        Observed class labels.
    C: array-like, shape = [n_classes, n_classes]
        Cost matrix.
    m_max: int
        Maximal number of hypothetically acquired labels.
    prior : int | array-like, shape [n_classes]
       Prior value for each class.

    Returns
    -------
    expected_cost_reduction: array-like, shape [n_samples]
        Expected cost reduction for given parameters.
    """
    n_classes = len(k_vec_list[0])
    n_samples = len(k_vec_list)

    # check cost matrix
    C = 1 - np.eye(n_classes) if C is None else np.asarray(C)

    # generate labelling vectors for all possible m values
    l_vec_list = np.vstack([gen_l_vec_list(m, n_classes) for m in range(m_max + 1)])
    m_list = np.sum(l_vec_list, axis=1)
    n_l_vecs = len(l_vec_list)

    # compute optimal cost-sensitive decision for all combination of k- and l-vectors
    k_l_vec_list = np.swapaxes(np.tile(k_vec_list, (n_l_vecs, 1, 1)), 0, 1) + l_vec_list
    y_hats = np.argmin(k_l_vec_list @ C, axis=2)

    # add prior to k-vectors
    prior = prior * np.ones(n_classes)
    k_vec_list = np.asarray(k_vec_list) + prior

    # all combination of k-, l-, and prediction indicator vectors
    combs = [k_vec_list, l_vec_list, np.eye(n_classes)]
    combs = np.asarray([list(elem) for elem in list(itertools.product(*combs))])

    # three factors of the closed form solution
    factor_1 = 1 / euler_beta(k_vec_list)
    factor_2 = multinomial(l_vec_list)
    factor_3 = euler_beta(np.sum(combs, axis=1)).reshape(n_samples, n_l_vecs, n_classes)

    # expected classification cost for each m
    m_sums = np.asarray(
        [factor_1[k_idx] * np.bincount(m_list, factor_2 * [C[:, y_hats[k_idx, l_idx]] @ factor_3[k_idx, l_idx]
                                                           for l_idx in range(n_l_vecs)]) for k_idx in
         range(n_samples)])

    # compute classification cost reduction as difference
    gains = np.zeros((n_samples, m_max)) + m_sums[:, 0].reshape(-1, 1)
    gains -= m_sums[:, 1:]

    # normalize classification cost reduction by number of hypothetical label acquisitions
    gains /= np.arange(1, m_max + 1)

    return np.max(gains, axis=1)

def gen_l_vec_list(m_approx, n_classes):
    """
    Creates all possible class labeling vectors for given number of hypothetically acquired labels and given number of
    classes.

    Parameters
    ----------
    m_approx: int
        Number of hypothetically acquired labels..
    n_classes: int,
        Number of classes

    Returns
    -------
    label_vec_list: array-like, shape = [n_labelings, n_classes]
        All possible class labelings for given parameters.
    """

    label_vec_list = [[]]
    label_vec_res = np.arange(m_approx + 1)
    for i in range(n_classes - 1):
        new_label_vec_list = []
        for labelVec in label_vec_list:
            for newLabel in label_vec_res[label_vec_res - (m_approx - sum(labelVec)) <= 1.e-10]:
                new_label_vec_list.append(labelVec + [newLabel])
        label_vec_list = new_label_vec_list

    new_label_vec_list = []
    for labelVec in label_vec_list:
        new_label_vec_list.append(labelVec + [m_approx - sum(labelVec)])
    label_vec_list = np.array(new_label_vec_list, int)

    return label_vec_list


def euler_beta(a):
    """
    Represents Euler beta function: B(a(i)) = Gamma(a(i,1))*...*Gamma(a_n)/Gamma(a(i,1)+...+a(i,n))

    Parameters
    ----------
    a: array-like, shape (m, n)
        Vectors to evaluated.

    Returns
    -------
    result: array-like, shape (m)
        Euler beta function results [B(a(0)), ..., B(a(m))
    """
    return np.exp(np.sum(gammaln(a), axis=1)-gammaln(np.sum(a, axis=1)))


def multinomial(a):
    """
    Computes Multinomial coefficient: Mult(a(i)) = (a(i,1)+...+a(i,n))!/(a(i,1)!...a(i,n)!)

    Parameters
    ----------
    a: array-like, shape (m, n)
        Vectors to evaluated.

    Returns
    -------
    result: array-like, shape (m)
        Multinomial coefficients [Mult(a(0)), ..., Mult(a(m))
    """
    return factorial(np.sum(a, axis=1))/np.prod(factorial(a), axis=1)


def probabilistic_al(classifier: BaseEstimator, X: modALinput,
                     n_instances: int = 1, random_tie_break: bool = False,
                     **pal_kwargs) -> Tuple[np.ndarray, modALinput]:
    """
    Uncertainty sampling query strategy. Selects the least sure instances for labelling.

    Args:
        classifier: The classifier for which the labels are to be queried.
        X: The pool of samples to query from.
        n_instances: Number of samples to be queried.
        random_tie_break: If True, shuffles utility scores to randomize the order. This
            can be used to break the tie when the highest utility score is not unique.
        **uncertainty_measure_kwargs: Keyword arguments to be passed for the uncertainty
            measure function.

    Returns:
        The indices of the instances from X chosen to be labelled;
        the instances from X chosen to be labelled.
    """
    prior = pal_kwargs.pop('prior', 0.001)
    n_classes = pal_kwargs.pop('prior', 3)

    X_labeled = classifier.X_training if classifier.X_training is not None else np.array([])
    y_labeled = classifier.y_training if classifier.y_training is not None else np.array([])
    X_cand = X if X is not None else np.array([])

    X = []
    for x in X_cand:
        X.append(x)
    for x in X_labeled:
        X.append(x)
    X = np.array(X)

    # Determine gamma with heuristic
    delta = np.sqrt(2) * 1e-6
    N = min(X.shape[0] * len(X), 200)
    D = X.shape[1]
    s = np.sqrt((2 * N * D) / ((N - 1) * np.log((N - 1) / delta**2)))
    gamma = 1 / (2 * s**2)

    # Calculate similarities
    clf_sim = PWC(len(X), gamma=gamma)
    clf_sim.fit(X, range(len(X)))
    sim = clf_sim.predict_proba(X, normalize=False)
    densities = np.sum(sim, axis=0)[:len(X_cand)]

    # Calculate gains with PWC
    clf = PWC(n_classes, gamma=gamma)
    clf.fit(X_labeled, y_labeled)
    k_vec = clf.predict_proba(X_cand, normalize=False)
    gains = densities * cost_reduction(k_vec, prior=prior, m_max=1)

    if not random_tie_break:
        query_idx = multi_argmax(gains, n_instances=n_instances)
    else:
        query_idx = shuffled_argmax(gains, n_instances=n_instances)

    return query_idx, X[query_idx]