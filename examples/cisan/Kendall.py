# This code is from https://github.com/sunblaze-ucb/Priv-PC-Differentially-Private-Causal-Graph-Discovery/blob/main/src/Kendalltaua.py

import numpy as np
from collections import namedtuple
from copy import copy
import warnings
from numpy import ma
from scipy.stats import mstats_basic
from scipy.stats._stats import _kendall_dis
import scipy.special as special
from scipy.stats import norm
from scipy.optimize import minimize
from scipy.integrate import quad
from causallearn.graph.Dag import Dag
from GraphUtils import reachable
from Utility import CIStatement

KendalltauResult = namedtuple("KendalltauResult", ("correlation", "pvalue"))


def _contains_nan(a, nan_policy="propagate"):
    policies = ["propagate", "raise", "omit"]
    if nan_policy not in policies:
        raise ValueError(
            "nan_policy must be one of {%s}" % ", ".join("'%s'" % s for s in policies)
        )
    try:
        # Calling np.sum to avoid creating a huge array into memory
        # e.g. np.isnan(a).any()
        with np.errstate(invalid="ignore"):
            contains_nan = np.isnan(np.sum(a))
    except TypeError:
        # If the check cannot be properly performed we fallback to omitting
        # nan values and raising a warning. This can happen when attempting to
        # sum things that are not numbers (e.g. as in the function `mode`).
        contains_nan = False
        nan_policy = "omit"
        warnings.warn(
            "The input array could not be properly checked for nan "
            "values. nan values will be ignored.",
            RuntimeWarning,
        )

    if contains_nan and nan_policy == "raise":
        raise ValueError("The input contains nan values")

    return (contains_nan, nan_policy)


def kendalltaua(x, y, initial_lexsort=None, nan_policy="propagate"):
    """
    Calculate Kendall's tau-a, a correlation measure for ordinal data.
    Kendall's tau is a measure of the correspondence between two rankings.
    Values close to 1 indicate strong agreement, values close to -1 indicate
    strong disagreement.  This is the 1945 "tau-b" version of Kendall's
    tau [2]_, which can account for ties and which reduces to the 1938 "tau-a"
    version [1]_ in absence of ties.
    Parameters
    ----------
    x, y : array_like
        Arrays of rankings, of the same shape. If arrays are not 1-D, they will
        be flattened to 1-D.
    initial_lexsort : bool, optional
        Unused (deprecated).
    nan_policy : {'propagate', 'raise', 'omit'}, optional
        Defines how to handle when input contains nan. 'propagate' returns nan,
        'raise' throws an error, 'omit' performs the calculations ignoring nan
        values. Default is 'propagate'. Note that if the input contains nan
        'omit' delegates to mstats_basic.kendalltau(), which has a different
        implementation.
    Returns
    -------
    correlation : float
       The tau statistic.
    pvalue : float
       The two-sided p-value for a hypothesis test whose null hypothesis is
       an absence of association, tau = 0.
    See also
    --------
    spearmanr : Calculates a Spearman rank-order correlation coefficient.
    theilslopes : Computes the Theil-Sen estimator for a set of points (x, y).
    weightedtau : Computes a weighted version of Kendall's tau.
    Notes
    -----
    The definition of Kendall's tau that is used is [2]_::
      tau = (P - Q) / sqrt((P + Q + T) * (P + Q + U))
    where P is the number of concordant pairs, Q the number of discordant
    pairs, T the number of ties only in `x`, and U the number of ties only in
    `y`.  If a tie occurs for the same pair in both `x` and `y`, it is not
    added to either T or U.
    References
    ----------
    .. [1] Maurice G. Kendall, "A New Measure of Rank Correlation", Biometrika
           Vol. 30, No. 1/2, pp. 81-93, 1938.
    .. [2] Maurice G. Kendall, "The treatment of ties in ranking problems",
           Biometrika Vol. 33, No. 3, pp. 239-251. 1945.
    .. [3] Gottfried E. Noether, "Elements of Nonparametric Statistics", John
           Wiley & Sons, 1967.
    .. [4] Peter M. Fenwick, "A new data structure for cumulative frequency
           tables", Software: Practice and Experience, Vol. 24, No. 3,
           pp. 327-336, 1994.
    Examples
    --------
    >>> from scipy import stats
    >>> x1 = [12, 2, 1, 12, 2]
    >>> x2 = [1, 4, 7, 1, 0]
    >>> tau, p_value = stats.kendalltau(x1, x2)
    >>> tau
    -0.47140452079103173
    >>> p_value
    0.2827454599327748
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    if x.size != y.size:
        raise ValueError(
            "All inputs to `kendalltau` must be of the same size, "
            "found x-size %s and y-size %s" % (x.size, y.size)
        )
    elif not x.size or not y.size:
        print("here1 is wrong!")
        return KendalltauResult(np.nan, np.nan)  # Return NaN if arrays are empty

    # check both x and y
    cnx, npx = _contains_nan(x, nan_policy)
    cny, npy = _contains_nan(y, nan_policy)
    contains_nan = cnx or cny
    if npx == "omit" or npy == "omit":
        nan_policy = "omit"

    if contains_nan and nan_policy == "propagate":
        return KendalltauResult(np.nan, np.nan)

    elif contains_nan and nan_policy == "omit":
        x = ma.masked_invalid(x)
        y = ma.masked_invalid(y)
        return mstats_basic.kendalltau(x, y)

    if initial_lexsort is not None:  # deprecate to drop!
        warnings.warn('"initial_lexsort" is gone!')

    def count_rank_tie(ranks):
        cnt = np.bincount(ranks).astype("int64", copy=False)
        cnt = cnt[cnt > 1]
        return (
            (cnt * (cnt - 1) // 2).sum(),
            (cnt * (cnt - 1.0) * (cnt - 2)).sum(),
            (cnt * (cnt - 1.0) * (2 * cnt + 5)).sum(),
            cnt,
        )

    size = x.size
    perm = np.argsort(y)  # sort on y and convert y to dense ranks
    x, y = x[perm], y[perm]
    y = np.r_[True, y[1:] != y[:-1]].cumsum(dtype=np.intp)

    # stable sort on x and convert x to dense ranks
    perm = np.argsort(x, kind="mergesort")
    x, y = x[perm], y[perm]
    x = np.r_[True, x[1:] != x[:-1]].cumsum(dtype=np.intp)

    dis = _kendall_dis(x, y)  # discordant pairs

    obs = np.r_[True, (x[1:] != x[:-1]) | (y[1:] != y[:-1]), True]
    cnt = np.diff(np.where(obs)[0]).astype("int64", copy=False)

    ntie = (cnt * (cnt - 1) // 2).sum()  # joint ties
    xtie, x0, x1, cntx = count_rank_tie(x)  # ties in x, stats
    ytie, y0, y1, cnty = count_rank_tie(y)  # ties in y, stats

    tot = (size * (size - 1)) // 2

    if xtie == tot or ytie == tot:
        return (KendalltauResult(np.nan, np.nan), cntx, cnty)

    # Note that tot = con + dis + (xtie - ntie) + (ytie - ntie) + ntie
    #               = con + dis + xtie + ytie - ntie
    con_minus_dis = tot - xtie - ytie + ntie - 2 * dis
    # tau = con_minus_dis / np.sqrt(tot - xtie) / np.sqrt(tot - ytie)
    tau = con_minus_dis / tot
    # Limit range to fix computational errors
    tau = min(1.0, max(-1.0, tau))

    # con_minus_dis is approx normally distributed with this variance [3]_
    var = (
        (size * (size - 1) * (2.0 * size + 5) - x1 - y1) / 18.0
        + (xtie * ytie) / (2.0 * size * (size - 1))
        + x0 * y0 / (9.0 * size * (size - 1) * (size - 2))
    )
    pvalue = special.erfc(np.abs(con_minus_dis) / np.sqrt(var) / np.sqrt(2))

    # Limit range to fix computational errors
    return KendalltauResult(min(1.0, max(-1.0, tau)), pvalue), cntx, cnty


def bincondKendall(data_matrix, x, y, k, **kwargs):
    s_size = len(k)
    row_size = data_matrix.shape[0]
    if s_size == 0:
        (tau, pval), _, _ = kendalltaua(data_matrix[:, x], data_matrix[:, y])
        tau = tau * np.sqrt(9.0 * row_size * (row_size - 1) / (4 * row_size + 10))
        pval = norm.sf(np.abs(tau))
        return tau, pval
    z = []
    for z_index in range(s_size):
        z.append(k.pop())
        pass

    dm_unique = np.unique(data_matrix[:, z], axis=0)
    sumwk = 0
    sumweight = 0
    tau = 0
    pval = 0
    for split_k in dm_unique:
        index = np.ones((row_size), dtype=bool)
        for i in range(s_size):
            index = (data_matrix[..., z[i]] == split_k[i]) & index

        new_dm = data_matrix[index, :]
        nk = new_dm.shape[0]
        if nk <= 2:
            continue
        (condtau, condpval), cntx, cnty = kendalltaua(new_dm[:, x], new_dm[:, y])
        if np.isnan(condpval):
            continue
        sigma0_sq = (4.0 * nk + 10) / (9.0 * nk * (nk - 1.0))
        tau += condtau / sigma0_sq
        sumwk += 1.0 / sigma0_sq

    tau /= np.sqrt(sumwk)
    pval = norm.sf(np.abs(tau))

    return tau, pval


def discondKendall(data_matrix, x, y, k, **kwargs):
    s_size = len(k)
    row_size = data_matrix.shape[0]
    if s_size == 0:
        (tau, pval), _, _ = kendalltaua(data_matrix[:, x], data_matrix[:, y])
        tau = tau * np.sqrt(9.0 * row_size * (row_size - 1) / (4 * row_size + 10))
        pval = norm.sf(np.abs(tau))
        return tau, pval
    z = []
    for z_index in range(s_size):
        z.append(k.pop())
        pass

    dm_unique = np.unique(data_matrix[:, z], axis=0)
    sumwk = 0
    sumweight = 0
    tau = 0
    pval = 0
    for split_k in dm_unique:
        index = np.ones((row_size), dtype=bool)
        for i in range(s_size):
            index = (data_matrix[..., z[i]] == split_k[i]) & index

        new_dm = data_matrix[index, :]
        nk = new_dm.shape[0]
        if nk <= 2:
            continue
        (condtau, condpval), cntx, cnty = kendalltaua(new_dm[:, x], new_dm[:, y])
        if np.isnan(condpval):
            continue
        sigma0_sq = (4.0 * nk + 10) / (9.0 * nk * (nk - 1.0))
        tau += condtau / sigma0_sq
        sumwk += 1.0 / sigma0_sq

    tau /= np.sqrt(sumwk)
    pval = norm.sf(np.abs(tau))

    return tau, pval


class DPKendalTau:
    def __init__(
        self,
        data: np.ndarray,
        eps: float = 1.5,
        delta=1e-3,
        alpha: float = 0.01,
        bias: float = 0.02,
        dag: Dag = None,
    ):

        self.data = data
        self.eps = eps
        self.delta = delta
        self.alpha = alpha
        self.bias = bias

        self.n = data.shape[0]
        self.node_size = data.shape[1]

        self.dag = dag
        self.oracle_cache = {}

        budget_split = 1.0 / 2.0
        self.eps1 = eps * budget_split

        def noise_scale(x):
            return np.sqrt(x) / np.log(x * (np.exp(self.eps1) - 1) + 1)

        self.q = max(
            min(1.0 / minimize(noise_scale, [0.5], tol=1e-2).x[0], 1), 1.0 / 20.0
        )
        print("Optimal Subsample Rate:", self.q)
        self.eps2 = eps - self.eps1
        S, _ = quad(
            lambda x: np.exp(-(x**2) / 2) / np.sqrt(2 * np.pi), 0, 6 / np.sqrt(self.n)
        )
        self.sigma1 = (
            2.0 * S / np.sqrt(self.q) / np.log((np.exp(self.eps1) - 1.0) / self.q + 1)
        )
        self.sigma2 = 2 * self.sigma1
        self.sigma3 = S / self.eps2
        self.T_hat = alpha - bias + np.random.laplace(0, self.sigma1)

        row_rand = np.arange(self.n)
        np.random.shuffle(row_rand)
        self.subsampled_data = self.data[row_rand[0 : int(self.n * self.q)]]

        self.ci_invoke_count = 0
        self.count = 0
        self.test_count = 0

        self.accuracy = [0, 0]

    def kendaltau_ci(self, x: int, y: int, z: set[int]):
        self.ci_invoke_count += 1
        self.test_count += 1
        v = np.random.laplace(0, self.sigma2)
        p_val = discondKendall(self.subsampled_data, x, y, copy(z))[1] + v
        # dependent
        if p_val < self.T_hat:
            rlt = False
        else:
            self.count += 1
            self.T_hat = self.alpha - self.bias + np.random.laplace(0, self.sigma1)
            row_rand = np.arange(self.n)
            np.random.shuffle(row_rand)
            self.subsampled_data = self.data[row_rand[0 : int(self.n * self.q)]]
            v = np.random.laplace(0, self.sigma3)
            p_val = discondKendall(self.subsampled_data, x, y, copy(z))[1] + v
            self.test_count += 1
            if p_val >= self.alpha:
                rlt = True
            else:
                rlt = False
        self.independence_oracle(x, y, z, rlt)
        return rlt

    def get_eps_prime(self):
        count = self.count
        delta = self.delta
        eps1 = self.eps1
        eps2 = self.eps2
        eps_prime1 = np.sqrt(2 * count * np.log(2 / delta)) * eps2 + count * eps2 * (
            np.exp(eps2) - 1
        )
        eps_prime2 = np.sqrt(2 * count * np.log(2 / delta)) * eps1 + count * eps1 * (
            np.exp(eps1) - 1
        )
        return eps_prime1 + eps_prime2

    def independence_oracle(self, x: int, y: int, z: set[int], rlt):
        f_z = frozenset(z)
        x_reachable: dict = self.oracle_cache.setdefault(x, {})
        if f_z in x_reachable:
            x_z_reachable = x_reachable[f_z]
        else:
            x_z_reachable: set = frozenset(reachable(x, z, self.dag))
        true_rlt = y not in x_z_reachable
        if true_rlt == rlt:
            self.accuracy[0] += 1
        self.accuracy[1] += 1

        print("Tested by Oracle:", CIStatement.createByXYZ(x, y, z, true_rlt))
        print("Tested by DPKT:", CIStatement.createByXYZ(x, y, z, rlt))
