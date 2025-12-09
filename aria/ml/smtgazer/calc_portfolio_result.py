from os import system
import time
import os
import sys
import subprocess
import datetime
from functools import partial
from multiprocessing import Pool
from os import popen


import json
import numpy as np
from copy import deepcopy

seed = [0,1,2,3,4,5,6,7,8,9]

def RunSeed(seed,key):

    dataset= key
    with open("./data/solverEval.json",'r', encoding='UTF-8') as f:
        solver_dict = json.load(f)

    with open("./data/" + str(dataset) + "Labels.json",'r', encoding='UTF-8') as f:
        par2_dict = json.load(f)

    c_num = 20
    if dataset == "Equality+LinearArith":
        c_num=2
    if dataset == "SyGuS":
        c_num=3
    with open("output/test_result_"+ str(key) +"_"+ str(seed) +"_"+str(c_num)+".json",'r', encoding='UTF-8') as f:
        test_dict = json.load(f)

    test_set = par2_dict['test']

    if dataset == "QF_Bitvec" or dataset == 'Equality+LinearArith' or dataset == 'QF_NonLinearIntArith':
        solverlist=list(solver_dict[dataset])
    elif dataset == "BMC" or dataset == "SymEx":
        solverlist = ['Bitwuzla','STP 2021.0','Yices 2.6.2 for SMTCOMP 2021','cvc5','mathsat-5.6.6','z3-4.8.11']
    elif dataset == "SyGuS":
        solverlist = ['cvc5','UltimateEliminator+MathSAT-5.6.6','smtinterpol-2.5-823-g881e8631','veriT','z3-4.8.11']

    key_set = list(test_set.keys())
    test_num = 0
    fail_num = 0

    if dataset == "Equality+LinearArith":
        dataplace = "ELA"
    elif dataset == "QF_Bitvec":
        dataplace = "QFBV"
    elif dataset == "QF_NonLinearIntArith":
        dataplace = "QFNIA"
    else:
        dataplace = dataset
    time = 0
    for _ in range(len(key_set)):
        par2list=test_set[key_set[_]]
        idx = np.argmin(par2list)

        if par2list[idx] == 2400:
            continue

        if  "./infer_result/"+str(dataset)+"/_data_sibly_sibyl_data_"+str(dataset)+"_"+str(dataset)+"_" + key_set[_].replace("/","_") + ".json" in test_dict.keys():
            portfolio_result = test_dict["./infer_result/"+str(dataset)+"/_data_sibly_sibyl_data_"+str(dataset)+"_"+str(dataset)+"_" + key_set[_].replace("/","_") + ".json"]
        elif  "./infer_result/"+str(dataplace)+"/_data_sibly_sibyl_data_Comp_non-incremental_" + key_set[_].replace("/","_") + ".json" in test_dict.keys():
            portfolio_result = test_dict["./infer_result/"+str(dataplace)+"/_data_sibly_sibyl_data_Comp_non-incremental_" + key_set[_].replace("/","_") + ".json"]
        else:
            continue
        slist = portfolio_result[0]
        x1,x2,x3,x4 = portfolio_result[1]
        output_idx = []
        for s in slist:
            output_idx.append(solverlist.index(s))

        test_num+=1
        tmp_time = 0
        if float(par2list[output_idx[0]]) <= x1:
            tmp_time+=par2list[output_idx[0]]
            time += tmp_time
            continue
        if float(par2list[output_idx[1]]) <= x2:
            tmp_time+=par2list[output_idx[1]]+x1
            time += tmp_time
            continue
        if float(par2list[output_idx[2]]) <= x3:
            tmp_time+=par2list[output_idx[2]]+x1+x2
            time += tmp_time
            continue
        if float(par2list[output_idx[3]]) <= x4:
            tmp_time+=par2list[output_idx[3]]+x1+x2+x3
            time += tmp_time
            continue
        time +=2400
        fail_num += 1
    return [seed,time,fail_num,test_num]

if __name__ == '__main__':
    key_set = ['Equality+LinearArith','QF_NonLinearIntArith',"QF_Bitvec","SyGuS","BMC","SymEx"]
    p = Pool(processes=10)
    for key in key_set:
        par2score = [0 for i in range(len(seed))]
        count_num = [0 for i in range(len(seed))]
        fail = [0 for i in range(len(seed))]

        partial_RunSeed = partial(RunSeed,key = key)
        ret = p.map(partial_RunSeed, seed)
        for l in ret:
            par2score[l[0]]=l[1]
            fail[l[0]] = l[2]
            count_num[l[0]] = l[3]
        print(key)
        print("PAR2: "+str(np.mean(par2score)/np.mean(count_num)))
        print("#UNK: "+str(np.mean(fail)))

    p.close()
    p.join()
