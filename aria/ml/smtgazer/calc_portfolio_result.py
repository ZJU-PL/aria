from functools import partial
from multiprocessing import Pool

import json
import numpy as np

# pylint: disable=redefined-outer-name,duplicate-code

seed = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


def run_seed(seed_val, category_key):

    dataset_name = category_key
    with open("./data/solverEval.json", "r", encoding="UTF-8") as f:
        solver_dict = json.load(f)

    with open(f"./data/{dataset_name}Labels.json", "r", encoding="UTF-8") as f:
        par2_dict = json.load(f)

    c_num = 20
    if dataset_name == "Equality+LinearArith":
        c_num = 2
    if dataset_name == "SyGuS":
        c_num = 3
    test_file = f"output/test_result_{category_key}_{seed_val}_" f"{c_num}.json"
    with open(test_file, "r", encoding="UTF-8") as f:
        test_dict = json.load(f)

    test_set = par2_dict["test"]

    if dataset_name in ("QF_Bitvec", "Equality+LinearArith", "QF_NonLinearIntArith"):
        solver_list = list(solver_dict[dataset_name])
    elif dataset_name in ("BMC", "SymEx"):
        solver_list = [
            "Bitwuzla",
            "STP 2021.0",
            "Yices 2.6.2 for SMTCOMP 2021",
            "cvc5",
            "mathsat-5.6.6",
            "z3-4.8.11",
        ]
    elif dataset_name == "SyGuS":
        solver_list = [
            "cvc5",
            "UltimateEliminator+MathSAT-5.6.6",
            "smtinterpol-2.5-823-g881e8631",
            "veriT",
            "z3-4.8.11",
        ]
    else:
        solver_list = []

    key_set = list(test_set.keys())
    test_num = 0
    fail_num = 0

    if dataset_name == "Equality+LinearArith":
        dataplace = "ELA"
    elif dataset_name == "QF_Bitvec":
        dataplace = "QFBV"
    elif dataset_name == "QF_NonLinearIntArith":
        dataplace = "QFNIA"
    else:
        dataplace = dataset_name
    total_time = 0
    for problem_idx, problem_key in enumerate(key_set):
        par2list = test_set[problem_key]
        idx = np.argmin(par2list)

        if par2list[idx] == 2400:
            continue

        key_alt1 = (
            f"./infer_result/{dataset_name}/_data_sibly_sibyl_data_"
            f"{dataset_name}_{dataset_name}_" + problem_key.replace("/", "_") + ".json"
        )
        key_alt2 = (
            f"./infer_result/{dataplace}/_data_sibly_sibyl_data_"
            f"Comp_non-incremental_" + problem_key.replace("/", "_") + ".json"
        )

        if key_alt1 in test_dict.keys():
            portfolio_result = test_dict[key_alt1]
        elif key_alt2 in test_dict.keys():
            portfolio_result = test_dict[key_alt2]
        else:
            continue
        slist = portfolio_result[0]
        x1, x2, x3, x4 = portfolio_result[1]
        output_idx = []
        for solver_name in slist:
            output_idx.append(solver_list.index(solver_name))

        test_num += 1
        tmp_time = 0
        if float(par2list[output_idx[0]]) <= x1:
            tmp_time += par2list[output_idx[0]]
            total_time += tmp_time
            continue
        if float(par2list[output_idx[1]]) <= x2:
            tmp_time += par2list[output_idx[1]] + x1
            total_time += tmp_time
            continue
        if float(par2list[output_idx[2]]) <= x3:
            tmp_time += par2list[output_idx[2]] + x1 + x2
            total_time += tmp_time
            continue
        if float(par2list[output_idx[3]]) <= x4:
            tmp_time += par2list[output_idx[3]] + x1 + x2 + x3
            total_time += tmp_time
            continue
        total_time += 2400
        fail_num += 1
    return [seed_val, total_time, fail_num, test_num]


if __name__ == "__main__":
    key_set = [
        "Equality+LinearArith",
        "QF_NonLinearIntArith",
        "QF_Bitvec",
        "SyGuS",
        "BMC",
        "SymEx",
    ]
    with Pool(processes=10) as p:
        for key in key_set:
            par2score = [0 for _ in range(len(seed))]
            count_num = [0 for _ in range(len(seed))]
            fail = [0 for _ in range(len(seed))]

            partial_run_seed = partial(run_seed, category_key=key)
            ret = p.map(partial_run_seed, seed)
            for result in ret:
                par2score[result[0]] = result[1]
                fail[result[0]] = result[2]
                count_num[result[0]] = result[3]
            print(key)
            print(f"PAR2: {np.mean(par2score) / np.mean(count_num)}")
            print(f"#UNK: {np.mean(fail)}")
