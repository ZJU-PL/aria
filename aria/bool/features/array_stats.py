import statistics as stats
import math
import scipy.stats as sci_stats

"""
File to control the computation and aggregation of statistics for lists of values.
"""


def get_stats(l):
    """
    Gets the four basic stats used for most features
    :param l: List to generate statistics from.
    :return: mean, co-efficient of variation, minimum and maximum.
    """
    mean = stats.mean(l)
    min_val = min(l)
    max_val = max(l)
    std = stats.pstdev(l)
    coefficient_of_variation = calc_coefficient_of_variation(mean, std)
    return mean, coefficient_of_variation, min_val, max_val


def get_stdev(l):
    """
    :param l: data
    :return: Population standard deviation
    """
    return stats.pstdev(l)


def calc_coefficient_of_variation(mean, std):
    """
    The coefficient of variation is a statistical measure of the relative dispersion of data points in a data series
    around the mean. https://en.wikipedia.org/wiki/Coefficient_of_variation
    :param mean:
    :param std:
    :return: Coefficient of variation
    """
    if std == 0:
        return 0
    else:
        return std / mean


def scipy_entropy_discrete(l, num_outcomes):
    """
    Create a probability distribution of l, and then get the entropy of that distribution
    :param l: Data
    :param num_outcomes: The total possible number of outcomes
    :return: Entropy of l
    """
    p = [0] * num_outcomes
    for elem in l:
        p[elem] += 1
    p = [x / len(l) for x in p]
    entropy = sci_stats.entropy(pk=p)
    return entropy


def scipy_entropy_continous(l, buckets=100):
    """
    Create a probability distribution of l, and then get the entropy of that distribution
    :param buckets:
    :param l: Data
    :return: Entropy of l
    """
    maxval = 1
    # set up probability distribution with number of buckets
    p = [0] * buckets
    for x in l:
        index = math.floor(x * (buckets / maxval))
        if index >= buckets:
            index = buckets - 1
        if index < 0:
            index = 0
        p[index] += 1

    # normalise
    p = [x / len(l) for x in p]
    entropy = sci_stats.entropy(pk=p)
    return entropy


# Legacy
def entropy_float_array(l, num, vals, maxval):
    """
    :param maxval:
    :param num:
    :param l: list of values (float, should be between 0 and 1)
    :param vals: size of the bins used (normally 100)
    For now, this will be based on the implementation from SATzilla
    (posneg-ratio-clause-entropy)
    What is int num -> number of clauses/variables -> length of input array
    What is int vals -> total possible outcomes (in this case they make buckets to form a probabilty distribution function)
    writeFeature("POSNEG-RATIO-CLAUSE-entropy",array_entropy(pos_frac_in_clause,numClauses,100,1));
    https://en.wikipedia.org/wiki/Entropy_(information_theory)
    """
    # seems to be off by 1 error in the sat implementation... you shouldn't need 101 buckets...
    p = [0] * (vals + 1)
    res = 0
    entropy = 0
    # based on the assumption that all values in l are floats between 0 and 1
    # set up the probability distribution distribution
    for t in range(num):
        # if l[t] == reserved_value ?:
        #   res ++
        #     continue

        # find the bucket it should go in
        # index = math.floor(l[t] / (maxval/vals))
        index = math.floor(l[t] * (vals / maxval))

        if index > vals:
            index = vals
        if index < 0:
            index = 0
        p[index] += 1

    for t in range(vals + 1):
        if p[t] != 0:
            # pval = p[t]/(num-res)
            pval = p[t] / num
            entropy += pval * math.log(pval)

    return -1 * entropy


def entropy_int_array(l, number_of_outcomes):
    """
    :param l: List of statistics (vcg variable/clause node degrees, )
    :param number_of_outcomes: upper bound on the maximum number of outcomes (e.g. for vcg clause node degree, it could have a maximum
    of the number of variables (the clauses contains all variables)).
    :return:
    vcg clause entropy = array_entropy(clause_array,numClauses,numActiveVars+1))
    Entropy of x  is H(X) = - sum (P(xi) * log(P(xi)))
    """
    # create the distribution buckets
    p = [0] * number_of_outcomes
    # res = 0
    entropy = 0
    # set up the probability distribution
    # Could also be in range of the list
    for elem in l:
        p[elem] += 1

    for t in range(number_of_outcomes):
        if p[t] != 0:
            pval = p[t] / len(l)
            # pval = p[t]/(num - res)
            entropy += pval * math.log(pval)

    return -1 * entropy
