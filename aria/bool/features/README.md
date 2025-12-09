# Feature Extraction for SAT Instances

This file documents the features that can be extracted by this software, and gives a brief explanation of each feature.

N.B. The CNF first undergoes pre-processing by SATELite, a preprocessor provided by MiniSAT. Implementation of
different,
and more up to date pre-processing algorithms is planned, and suggestions are welcome. It is also possible to perform
the
feature extraction on non pre-processed CNF, although many SATzilla features make assumptions based on conditions that
the preprocessing ensures.

## SATzilla Paper Features

Numbers refer to the feature number within [1].

### Size Features

- Number of clauses: _c_ (1.)
- Number of variables: _v_ (2.)
- Ratio of clauses to variables c/v: _clauses_vars_ratio_ (3.)

### Variable-Clause Graph Features

*A variable-clause graph (VCG) is a bipartite graph with a node for each variable, a node for each clause,
and an edge between them whenever a variable occurs in a clause.*

- Variable nodes degree statistics: mean, coefficient of variation, min, max and entropy. *vcg_var_**statistic*** (
  4-8. )
- Clause nodes degree statistics: mean, coefficient of variation, min, max and entropy. *vcg_clause_**statistic*** (
  9-13. )

### Variable Graph Features

**A variable graph (VG) has a node for each variable, and an edge between variables that occur together in at least one
clause**

- Nodes degree statistics: mean , coefficient of variation, min and max. *vg_**statistic*** (14-17. )

### Balance Features

- Ratio of positive and negative literals in each clause: mean, variation coefficient and entropy (see extra notes).
  *pnc_ratio_**statistic*** (18-20.)
  n.b. The _ratio_ is not actually a ratio, but a measure of the offset/bias between the positive and negative values,
  as calculated by: 2 * abs(0.5 - pos/(pos + neg))
- Ratio of positive and negative occurrences of each variable: mean, coefficient of variation, min, max and entropy.
  *pnv_ratio_**statistic*** (21-25. )
- Fraction of binary and ternary clauses: Fraction of clauses that have 2 literals over total number of clauses,
  fraction of clauses with 2 or 3 literals, fraction of clauses with 3 literals, over total number of clauses.
  _binary_ratio_, _ternary+_, _ternary_ratio_, (respectively) (26-27.)

### Proximity to Horn Formula

**A clause is a Horn clause if it contains at most one positive literal**

- Fraction of Horn clauses: _hc_fraction_ (28.)
- Number of occurrences in a Horn clause for each variable: mean, coefficient of variation, minimum, maximum, entropy.
  *hc_var_**statistic*** The number of times a variable appears in all horn clauses. (29-33.)

### DPLL Probing Features

**DPLL (Davis-Putnam-Logemann-Loveland) algorithm is a backtracking algorithm for solving CNF SAT problems.**

- Number of unit propagations: computed at depths 1, 4, 16, 64 and 256. *unit_props_at_depth_**depth*** (34-38.)
- Search space size estimate: mean depth to contradiction, estimate of the log of number of nodes.
  _mean_depth_to_contradiction_over_vars_, _estimate_log_number_nodes_over_vars_ (39-40.)

### Local Search Probing Features

**The SAPS and GSAT algorithms for solving SAT problems are run up to 10000 times,
(or as many times within a 2 second time limit) on the cnf. Various statistics over the runs are calculated**

- Number of steps to the best local minimum in a run: mean, median, 10th and 90th percentiles for SAPS and GSAT. (
  41-44.)
- Average improvement to best in a run: mean improvement to best solution for SAPS and GSAT. (45.)
- Fraction of improvement due to first local minimum: mean for SAPS and GSAT. (46-47.)
- Coefficient of variation of the number of unsatisfied clauses in each local minimum. (48.)

**alg** = saps or gsat
***alg**_BestSolution_Mean*, ***alg**_BestSolution_CoeffVariance*, ***alg**_FirstLocalMinStep_Mean*,
***alg**_FirstLocalMinStep_CoeffVariance*, ***alg**_FirstLocalMinStep_Median*, ***alg**_FirstLocalMinStepQ_.10*,
***alg**_FirstLocalMinStep_Q.90*, ***alg**_BestAvgImprovement_Mean*, ***alg**_BestAvgImprovement_CoeffVariance*,
***alg**_FirstLocalMinRatio_Mean*, ***alg**_FirstLocalMinRatio_CoeffVariance*, ***alg**_EstACL_Mean*

### Features from Structure Features for SAT Instances Classification (Ansotegui)

4 structure features

- alpha_v: powerlaw exponent (_variable_alpha_)
- Q: modularity (for VIG) (_vig_modularity_)
- d: fractal dimension for VIG (_vig_d_poly_)
- d_b: fractal dimension for CVIG (_cvig_db_poly_)

TODO: better explanation of all features, starting with variable occurrences alpha
A variable incidence graph (VIG): Set of vertexes is the set of boolean variables - weights assigned to edges as follows
W(x, y) = sum (1/(|c| choose 2)) where x and y are an element of c

Clause variable incidence graph (CVIG): Set of vertexes is set of variables and clauses, weight function:
w(x, c) = 1/|c| if x elem c
0 otherwise.
(signs of literals not considered)

Graphs are the same as the variable graph and variable clause graph from SATzilla, just with weights, which are,
incidentally, unused.

- Scale free structure - power law distribution. Estimation computed by method of Maximum likelihood
- Modularity(Q) of VIG (louvain method) of the best partition.
- Fractal dimensions of VIG and CVIG - computed by interpolating log N(r) vs log r, as N(r) ~ r^-d.
  N(r) is the estimate of the minimum number of circles with radius r that cover the graph.
  Estimated using the Burning by node degree algorithm.

Please refer to [2] for a more detailed and theoretical description of the features.

See [GraphFeatSat](https://www.ugr.es/~jgiraldez/)
and [code](https://www.ugr.es/~jgiraldez/download/graph_features_sat_v_2_2.tar.gz) for original implementation.

### Features from New CNF Features and Formula Classification (Alfonso)

Features

- 8 graphs with weights
    - Clause Variable: *c_nd_p_**statistic***, *c_nd_n_**statistic***, *v_nd_p_**statistic***, *v_nd_n_**statistic***, (
      for positive and negative, clause and variable node statistics, respectively) (for positive variables, and
      negative variables, otherwise the same as previously described)
    - Variable graph: *vg_al_**statistic*** (Does not consider polarity of literals, and with weights as the number of
      common clauses)
    - Clause graph: *cg_al_**statistic*** (Same as previously described, with weights the size of the intersection of
      clauses)
    - Resolution graph: *rg_**statistic*** (Connects clauses when they produce a non-tautological resolvant, weights as
      2^-(|C_i union C_j| - 2))
    - Binary Implication graph (BIG): *big_**statistic*** (Edges correspond to the binary clauses in the current
      formula, and if the
      corresponding variable is implicated by the variables in that clause) (clauses as vertices, edge between clauses
      iff
      | C_i intersection (not C_j) | = 1) (there is one literal in clause i that also appears as negated in clause j),
      with weights 2 ^ -(|C_i union C_j| -2) (degrees, weights). All edges in the graph correspond to the
      binary clauses in the current formula E = {(a, b), (not b, not a) | {not a, b} elem F}. Sequence with the degree
      of each node is used.
    - AND, BAND, EXO graph: *and_**statistic***, *band_**statistic***, *exo_**statistic*** (These naive encoded
      restraints create 3 graphs, extracted from the BIG).
      Creation of the graphs is based on theoretical description of them and pseudocode as provided in [3].
- Recursive weight heuristic: *rwh_**iteration**_**statistic*** The heuristic provides a score for each literal x that
  represents the tendency whether x is present in a model of the formula F.
  This is computed for 3 iterations. This was based on the implementation of the RISS sat solver.

Sequences from graphs are the weights of edges and degrees ofnodes. Statistics from these are calculated (Similiar as to
SATzilla, with additional statistics such as quantiles, 0 weight occurrences and the number times in which the modal
value occurs.

Please refer to [3] for a more detailed and theoretical description of the features.

Statistics for these graphs: node_min, node_max, node_mode, node_mean, node_std, node_zeros, node_entropy, node_q1,
node_q2, node_q3, node_val_rate,
weights_min, weights_max, weights_mode, weights_mean, weights_std, weights_zeros, weights_entropy, weights_q1,
weights_q2, weights_q3, weights_val_rate,
**where** node represents the degrees of the nodes, and weights represents the weights of the edges.

### Extra Notes on Aggregation and Statistical References

- Coefficient of variation (variation coefficient in paper):
  The coefficient of variation represents the ratio of the standard deviation to the mean
- Entropy: Shannon entropy (https://en.wikipedia.org/wiki/Entropy_(information_theory)). A distribution of the array is
  made, from which the entropy is calculated.
- The degree of a node is the number of edges incident (connected) to that node.