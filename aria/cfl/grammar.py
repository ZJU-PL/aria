"""
Context-Free Grammar Parser and Converter

This module provides comprehensive functionality for parsing and converting
Extended Backus-Naur Form (EBNF) grammars to Binary Normal Form (BNF).
It is specifically designed for use in CFL-reachability analysis.

Key Features:
- Converts EBNF to BNF by introducing new non-terminals
- Handles special constructs: Kleene star (*), optional (?), grouping ()
- Converts long productions into binary form (at most two symbols on RHS)
- Identifies epsilon (ε) productions for nullable variables
- Maintains production rules in a structured dictionary format

The conversion process follows standard grammar normalization techniques:
1. Handle repetition operators (* and ?)
2. Remove grouping parentheses
3. Convert to binary normal form
4. Extract epsilon productions

Author: aria team
"""

import re
import copy
from typing import Dict, List, Any, Union, Tuple


class Grammar:
    """
    A context-free grammar parser and converter for CFL-reachability analysis.

    This class handles the parsing and normalization of context-free grammars
    from EBNF format to BNF format. It supports various grammar constructs
    and ensures the resulting grammar is suitable for CFL-reachability algorithms.

    Attributes:
        new_terminal_subscript (int): Counter for generating new non-terminal names.
        edge_pair_dict (Dict[str, List[List[str]]]): Dictionary mapping non-terminals
                                                   to their production rules.
        epsilon (List[str]): List of non-terminals that can derive epsilon.
        search_pattern (re.Pattern): Compiled regex for parsing grammar files.
        filename (str): Path to the input grammar file.

    Example:
        >>> grammar = Grammar("example_grammar.txt")
        >>> print(grammar.keys())  # List of non-terminals
        >>> print(grammar.epsilon)  # List of nullable variables
    """

    def __init__(self, filename: str) -> None:
        self.new_terminal_subscript: int = 0
        # list contain all the edge that has epsilon symbol in right hand side
        self.edge_pair_dict: Dict[str, List[List[str]]] = {}
        self.epsilon: List[str] = []
        self.search_pattern = re.compile(r"Start:([\s\S]*)Productions:([\s\S]*)")
        self.filename: str = filename
        self.grammar(self.filename)

    def keys(self) -> List[str]:
        """
        Get all non-terminal symbols in the grammar.

        Returns:
            List[str]: List of non-terminal symbols (left-hand sides of productions).
        """
        return list(self.edge_pair_dict.keys())

    def items(self) -> List[Tuple[str, List[List[str]]]]:
        """
        Get all production rules as (non-terminal, productions) pairs.

        Returns:
            List[Tuple[str, List[List[str]]]]: List of tuples where each tuple
                                             contains (non_terminal, list_of_productions).
        """
        return list(self.edge_pair_dict.items())

    def ebnf_file_reader(self, filename: str) -> List[str]:
        search_pattern = self.search_pattern
        with open(filename, "r", encoding="utf-8") as f:
            string = f.read()
        match_instance = search_pattern.search(string)
        if match_instance is None:
            raise Exception("The form of ebnf is not correct.")
        production_rules = match_instance.group(2).split(";")
        return production_rules

    def ebnf_grammar_loader(
        self, production_rules: List[str]
    ) -> Dict[str, List[List[str]]]:
        # paser the string to dict datastructure
        grammar: Dict[str, List[List[str]]] = {}
        for rule in production_rules:
            _ = rule.split("->")
            head, LHS = _[0].strip(), _[1]
            if head not in grammar:
                grammar[head] = []
            for rule in LHS.split("|"):
                rule = rule.split()
                grammar[head].append(rule)
        return grammar

    def ebnf_bracket_match(self, rule: List[str], i_position: int) -> int:
        index = i_position
        while index >= 0:
            if rule[index] == "(":
                return index
            index -= 1
        raise Exception("Ebnf form is not correct.")

    def num_generator(self) -> int:
        self.new_terminal_subscript += 1
        return self.new_terminal_subscript

    # Convert every repetition * or { E } to a fresh non-terminal X and add
    # X = $\epsilon$ | X E
    # Convert every option ? [ E ] to a fresh non-terminal X and add
    # X = $\epsilon$ | E.
    def ebnf_sign_replace(
        self, grammar: Dict[str, List[List[str]]], sign: str
    ) -> Dict[str, List[List[str]]]:
        if sign not in ("?", "*"):
            raise Exception("Only accept ? or *")
        # select * position
        new_rule_checker: Dict[str, str] = {}
        for head in grammar:
            for rule in grammar[head]:
                i = 0
                while i < len(rule):
                    if rule[i] == sign:
                        if i == 0:
                            raise Exception("Ebnf form is not correct!")
                        if rule[i - 1] != ")":
                            repetition_start = i - 1
                        else:
                            repetition_start = self.ebnf_bracket_match(rule, i)
                        repetition = " ".join(rule[repetition_start : i + 1])
                        if repetition in new_rule_checker:
                            rule[repetition_start : i + 1] = [
                                new_rule_checker[repetition]
                            ]
                        else:
                            X = f"X{self.num_generator()}"
                            rule[repetition_start : i + 1] = [X]
                            new_rule_checker[repetition] = X
                        i = repetition_start
                    i += 1
        for repetition in new_rule_checker:
            new_nonterminal = new_rule_checker[repetition]
            if sign == "*":
                temp_list = [new_nonterminal]
            else:
                temp_list = []
            temp_list.extend(repetition[0:-1].split())
            grammar[new_nonterminal] = [["ε"], temp_list]
        return grammar

    # Convert every group ( E ) to a fresh non-terminal X and add
    # X = E.
    def ebnf_group_replace(
        self, grammar: Dict[str, List[List[str]]]
    ) -> Dict[str, List[List[str]]]:
        for head in grammar:
            for rule in grammar[head]:
                for element in rule:
                    if element in ("(", ")"):
                        rule.remove(element)
        return grammar

    def check_head(
        self, grammar: Dict[str, List[List[str]]], rule: List[str]
    ) -> Union[str, bool]:
        for in_head in grammar:
            for in_rule in grammar[in_head]:
                if rule == in_rule:
                    return in_head
        return False

    def ebnf_BIN(
        self, grammar: Dict[str, List[List[str]]]
    ) -> Dict[str, List[List[str]]]:
        new_grammar: Dict[str, List[List[str]]] = {}
        for head in grammar:
            for rule in grammar[head]:
                if len(rule) >= 3:
                    long_rule = copy.copy(rule)
                    first = long_rule.pop(0)
                    rule.clear()
                    rule.append(first)
                    # check whether has this rule
                    x_var = self.check_head(new_grammar, long_rule)
                    if x_var is False:
                        x_var = self.check_head(grammar, long_rule)
                    if x_var:
                        rule.append(x_var)
                        break
                    x_var = f"X{self.num_generator()}"
                    rule.append(x_var)
                    new_grammar[x_var] = []
                    temp = copy.copy(long_rule)
                    if len(long_rule) == 2:
                        new_grammar[x_var].append(temp)
                        long_rule.clear()
                    else:
                        first = long_rule.pop(0)
                        rhx = self.check_head(new_grammar, long_rule)
                        if rhx is False:
                            rhx = self.check_head(grammar, long_rule)
                        if rhx is False:
                            rhx = f"X{self.num_generator()}"
                        new_grammar[x_var].append([first, rhx])
                    while len(long_rule) >= 2:
                        if len(long_rule) == 2:
                            new_grammar[rhx] = []
                            new_grammar[rhx].append(long_rule)
                            break
                        first = long_rule.pop(0)
                        print(f"long{long_rule}")
                        # check whether has this rule
                        x_var = rhx
                        rhx = self.check_head(new_grammar, long_rule[1:])
                        if rhx is False:
                            rhx = self.check_head(grammar, long_rule[1:])
                        if rhx is False:
                            rhx = f"X{self.num_generator()}"
                        temp_rule = [first, rhx]
                        new_grammar[x_var] = []
                        new_grammar[x_var].append(temp_rule)

        for new_head, new_rules in new_grammar.items():
            grammar[new_head] = new_rules
        return grammar

    def ebnf_bnf_normal_convertor(self, filename: str) -> Dict[str, List[List[str]]]:
        production_rules = self.ebnf_file_reader(filename)
        grammar = self.ebnf_grammar_loader(production_rules)
        grammar = self.ebnf_sign_replace(grammar, "*")
        grammar = self.ebnf_sign_replace(grammar, "?")
        grammar = self.ebnf_group_replace(grammar)
        grammar = self.ebnf_BIN(grammar)
        return grammar

    def grammar(self, filename: str) -> None:
        self.edge_pair_dict = self.ebnf_bnf_normal_convertor(filename)
        for left_variable in self.keys():
            for rule in self.edge_pair_dict[left_variable]:
                if rule == ["ε"]:
                    self.epsilon.append(left_variable)
