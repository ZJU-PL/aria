"""
Compute SAT features following SATZilla
"""
from typing import Dict, List, Any

from aria.bool.features import array_stats, balance_features, graph_features


def write_stats(l: List[float], name: str, features_dict: Dict[str, Any]) -> None:
    """
    Write statistics (mean, coeff, min, max) to features dictionary.
    :param l: List of float values
    :param name: Base name for the features
    :param features_dict: Dictionary to store features
    """
    l_mean, l_coeff, l_min, l_max = array_stats.get_stats(l)

    features_dict[name + "_mean"] = l_mean
    features_dict[name + "_coeff"] = l_coeff
    features_dict[name + "_min"] = l_min
    features_dict[name + "_max"] = l_max


def write_entropy_discrete(
    l: List[int], number_outcomes: int, name: str,
    features_dict: Dict[str, Any]
) -> None:
    """
    Write discrete entropy to features dictionary.
    :param l: List of integer values
    :param number_outcomes: Number of possible outcomes
    :param name: Base name for the feature
    :param features_dict: Dictionary to store features
    """
    entropy = array_stats.scipy_entropy_discrete(l, number_outcomes)
    features_dict[name + "_entropy"] = entropy


def write_entropy_continous(
    l: List[float], name: str, features_dict: Dict[str, Any]
) -> None:
    """
    Write continuous entropy to features dictionary.
    :param l: List of float values
    :param name: Base name for the feature
    :param features_dict: Dictionary to store features
    """
    entropy = array_stats.scipy_entropy_continous(l)
    features_dict[name + "_entropy"] = entropy


def compute_base_features(
    clauses: List[List[int]],
    c: int,
    v: int,
    num_active_vars: int,
    num_active_clauses: int
) -> Dict[str, float]:
    features_dict = {"c": num_active_clauses, "v": num_active_vars,
                     "clauses_vars_ratio": num_active_clauses / num_active_vars,
                     "vars_clauses_ratio": num_active_vars / num_active_clauses}
    # 1-3

    c = num_active_clauses
    v = num_active_vars

    # Variable Clause Graph features
    vcg_v_node_degrees, vcg_c_node_degrees = graph_features.create_vcg(clauses, c, v)
    # variable node degrees divided by number of active clauses
    vcg_v_node_degrees_norm = [x / c for x in vcg_v_node_degrees]
    # 4-8
    write_stats(vcg_v_node_degrees_norm, "vcg_var", features_dict)
    write_entropy_discrete(vcg_v_node_degrees, c + 1, "vcg_var", features_dict)
    # write_entropy(vcg_v_node_degrees, "vcg_var", features_dict, v, c)

    # clause node degrees divided by number of active variables
    vcg_c_node_degrees_norm = [x / v for x in vcg_c_node_degrees]
    # 9-13
    write_stats(vcg_c_node_degrees_norm, "vcg_clause", features_dict)
    write_entropy_discrete(vcg_c_node_degrees, v + 1, "vcg_clause", features_dict)
    # write_entropy(vcg_c_node_degrees, "vcg_clause", features_dict, c, v)

    # Variable graph features
    vg_node_degrees = graph_features.create_vg(clauses)
    # 14-17
    # variable node degrees divided by number of active clauses
    vg_node_degrees_norm = [x / c for x in vg_node_degrees]

    write_stats(vg_node_degrees_norm, "vg", features_dict)

    # Balance features
    (_, pos_neg_clause_balance,
     _, pos_neg_variable_balance,
     num_binary_clauses, num_ternary_clauses, num_horn_clauses,
     horn_clause_variable_count) = \
        balance_features.compute_balance_features(clauses, c, v)
    # 18-20
    write_stats(pos_neg_clause_balance, "pnc_ratio", features_dict)
    # write_entropy_float(pos_neg_clause_balance, "pnc_ratio", features_dict, c)
    write_entropy_continous(pos_neg_clause_balance, "pnc_ratio", features_dict)

    # 21-25
    write_stats(pos_neg_variable_balance, "pnv_ratio", features_dict)
    # write_entropy_float(pos_neg_variable_balance, "pnv_ratio", features_dict, num_active_vars)
    write_entropy_continous(pos_neg_variable_balance, "pnv_ratio", features_dict)

    features_dict["pnv_ratio_stdev"] = array_stats.get_stdev(pos_neg_variable_balance)
    # 26-27
    features_dict["binary_ratio"] = num_binary_clauses / c
    features_dict["ternary_ratio"] = num_ternary_clauses / c
    features_dict["ternary+"] = (num_binary_clauses + num_ternary_clauses) / c
    # 28
    features_dict["hc_fraction"] = num_horn_clauses / c
    # 29-33
    horn_clause_variable_count_norm = [x / c for x in horn_clause_variable_count]
    write_stats(horn_clause_variable_count_norm, "hc_var", features_dict)
    # write_entropy(horn_clause_variable_count, "hc_var", features_dict, v, c)
    write_entropy_discrete(horn_clause_variable_count, c + 1, "hc_var", features_dict)

    return features_dict


# legacy


def write_entropy(
    l: List[int], name: str, features_dict: Dict[str, Any],
    c: int, number_of_outcomes: int
) -> None:
    """
    Write entropy (legacy function).
    :param l: List of integer values
    :param name: Base name for the feature
    :param features_dict: Dictionary to store features
    :param c: Number of clauses (unused)
    :param number_of_outcomes: Number of possible outcomes
    """
    entropy = array_stats.entropy_int_array(l, number_of_outcomes + 1)
    print("saten", entropy)
    features_dict[name + "_entropy"] = entropy


def write_entropy_float(
    l: List[float], name: str, features_dict: Dict[str, Any],
    num: int, buckets: int = 100, maxval: int = 1
) -> None:
    """
    Write float entropy (legacy function).
    scipy has an implementation for shannon entropy
    (https://docs.scipy.org/doc/scipy/reference/generated/
    scipy.stats.entropy.html), could be something to look into changing to
    :param l: List of float values
    :param name: Base name for the feature
    :param features_dict: Dictionary to store features
    :param num: Number of values
    :param buckets: Number of buckets
    :param maxval: Maximum value
    """
    entropy = array_stats.entropy_float_array(l, num, buckets, maxval)
    print("saten", entropy)
    features_dict[name + "_entropy"] = entropy
