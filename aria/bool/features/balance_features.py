def compute_balance_features(clauses, c, v):
    """
    Computes the balance features
    :param clauses: List of clauses of the cnf
    :param c: number of clauses
    :param v: number of variables
    :return:
    """
    variables_pos_count = [0] * v
    variables_neg_count = [0] * v
    pos_neg_variable_ratios = []
    pos_neg_variable_balance = []
    # positive and negative counts of variables (literal instances)

    pos_neg_clause_ratios = []
    pos_neg_clause_balance = []
    num_horn_clauses = 0
    num_binary_clauses = 0
    num_ternary_clauses = 0

    horn_clause_variable_count = [0] * v

    for clause in clauses:
        if len(clause) == 2:
            num_binary_clauses += 1
        if len(clause) == 3:
            num_ternary_clauses += 1
        pos = 0
        neg = 0

        for literal in clause:
            if literal < 0:
                neg += 1
                literal = abs(literal)
                variables_neg_count[literal - 1] += 1
            else:
                pos += 1
                variables_pos_count[literal - 1] += 1

        if neg == 0:
            ratio = 1
        else:
            ratio = pos / neg
        pos_neg_clause_ratios.append(ratio)

        pos_neg_clause_balance.append(2.0 * abs(0.5 - (pos / (pos + neg))))

        # clause is a horn clause if it has at most 1 positive literal
        if pos <= 1:
            num_horn_clauses += 1

            for literal in clause:
                horn_clause_variable_count[abs(literal) - 1] += 1

    # calculate the ratio of positive and negative literals
    # per variable
    for i in range(v):
        pos_instances = variables_pos_count[i]
        neg_instances = variables_neg_count[i]
        if neg_instances == 0:
            vi_ratio = 1
        else:
            vi_ratio = pos_instances / neg_instances

        pos_neg_variable_ratios.append(vi_ratio)

        if (pos_instances == 0) and (neg_instances == 0):
            pos_neg_variable_balance.append(0.0)
        else:
            pos_neg_variable_balance.append(2.0 * abs(0.5 - (pos_instances / (pos_instances + neg_instances))))

    # dictionary could be a cleaner way to format and return the results
    return pos_neg_clause_ratios, pos_neg_clause_balance, pos_neg_variable_ratios, pos_neg_variable_balance, \
        num_binary_clauses, num_ternary_clauses, num_horn_clauses, horn_clause_variable_count
