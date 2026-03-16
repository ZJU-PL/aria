Linear Teamporal Logic
=======================


Linear Temporal Logic (LTL) is a formalism for expressing properties of reactive systems. It is a modal logic, i.e., it uses modal operators to express the temporal relationships between propositions. LTL is used in formal verification of software and hardware systems, and is supported by model checkers such as SPIN and NuSMV.

LTL is a linear-time temporal logic, which means that it describes the behavior of a system as a sequence of states, and the temporal operators in LTL operate on this sequence. The basic temporal operators in LTL are:

- **X** (Next): The operator X p is true in a state s if p is true in the next state s'.
- **G** (Globally): The operator G p is true in a state s if p is true in all states reachable from s.
- **F** (Finally): The operator F p is true in a state s if p is true in some state reachable from s.
- **U** (Until): The operator p U q is true in a state s if there is a state s' reachable from s such that q is true in s', and for all states s'' reachable from s' until s', p is true.

LTL formulas are built using propositional logic operators (AND, OR, NOT) and the temporal operators. For example, the formula G (p -> F q) expresses that q is eventually true after p becomes true in all states.
