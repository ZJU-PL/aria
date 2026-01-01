"""The interface"""

import json
from aria.bool.features import parse_cnf, active_features, base_features
from aria.bool.features.dpll import DPLLProbing


def write_features_to_json(results_dict):
    """
    Write features dictionary to JSON file.
    :param results_dict: Dictionary of features to write
    """
    with open("features.json", "w", encoding="utf-8") as f:
        json.dump(results_dict, f)


class SATInstance:
    """
    Class to hold the methods for generating features from a cnf. This class handles the parsing of the cnf file into
    data structures necessary to the perform feature extraction. Then the various features can be generated, and are
    stored in the features dictionary.
    """

    def __init__(self, input_cnf: str, verbose=False):
        self.verbose = verbose
        #  However, we may consider using the one in aria.bool.cnfsimplifier
        self.preprocess = False
        # TODO: allow using numerical clauses, cnf strings, and pysat CNF objects to
        #  initialized this object
        self.path_to_cnf = input_cnf  # path to a cnf file

        # satelite preprocessing
        # n.b. satelite only works on linux, mac no longer supports 32 bit binaries...

        # parse the cnf file
        if self.verbose:
            print("Parsing cnf file")
        self.clauses, self.c, self.v = parse_cnf.parse_cnf_file(self.path_to_cnf)

        if self.v == 0 or self.c == 0:
            self.solved = True
            return
        self.solved = False

        # computed with active features
        # These change as they are processed with dpll probing algorithms
        self.num_active_vars = 0
        self.num_active_clauses = 0
        # states and lengths of the clauses
        self.clause_states = []
        self.clause_lengths = []
        # array of the length of the number of variables, containing the number of active clauses,
        # and binary clauses that each variable contains
        self.num_active_clauses_with_var = []
        self.num_bin_clauses_with_var = []
        # stack of indexes of the clauses that have 1 literal
        self.unit_clauses = []

        # all of the clauses that contain a positive version of this variable
        self.clauses_with_positive_var = []
        self.clauses_with_negative_var = []
        # used for dpll operations, perhaps better to keep them in a dpll class...

        self.var_states = []

        self.features_dict = {}

        # necessary for unit propagation setup
        if self.verbose:
            print("Parsing active features")
        self.parse_active_features()

        # Do first round of unit prop to remove all unit clauses
        self.dpll_prober = DPLLProbing(self)
        if self.verbose:
            print("First round of unit propagation")
        self.dpll_prober.unit_prop(0, 0)

    def clauses_with_literal(self, literal):
        """
        Returns a list of clauses that contain the literal
        :param literal: Literal value (positive or negative)
        :return: List of clause indices containing the literal
        """
        if literal > 0:
            return self.clauses_with_positive_var[literal]
        return self.clauses_with_negative_var[abs(literal)]

    def parse_active_features(self):
        """
        Parse and compute active features for the SAT instance.
        """
        # self.num_active_vars, self.num_active_clauses, self.clause_states,
        # self.clauses, self.num_bin_clauses_with_var, self.var_states =\
        active_features.get_active_features(self, self.clauses, self.c, self.v)

    def gen_basic_features(self):
        """
        Generates the basic features (Including but not limited to 1-33 from the satzilla paper).
        """
        if self.verbose:
            print("Generating basic features")

        base_features_dict = base_features.compute_base_features(
            self.clauses, self.c, self.v, self.num_active_vars, self.num_active_clauses
        )
        self.features_dict.update(base_features_dict)

    def gen_dpll_probing_features(self):
        """
        Generates the dpll probing features (34-40 from the satzilla paper).
        """
        if self.verbose:
            print("DPLL probing")

        self.dpll_prober.unit_propagation_probe(False)

        self.dpll_prober.search_space_probe()

        self.features_dict.update(self.dpll_prober.unit_props_log_nodes_dict)

    def gen_local_search_probing_features(self):
        """
        Generates the local search probing features (including but not limited to 41-48 from the satzilla paper).
        """
        # also doesnt seem to fully work on osx.
        if self.verbose:
            print("Local search probing with SAPS and GSAT")

        raise NotImplementedError()

    def display_results(self):
        """
        Display all computed features.
        """
        for ele in self.features_dict.items():
            print(ele[0], ele[1])

    def write_results(self):
        """
        Write computed features to JSON file.
        """
        write_features_to_json(self.features_dict)


def get_base_features(cnf_path):
    sat_inst = SATInstance(cnf_path)
    sat_inst.parse_active_features()
    sat_inst.display_results()

    sat_inst.gen_basic_features()
    sat_inst.display_results()

    # sat_inst.gen_dpll_probing_features()
    # sat_inst.display_results()


def demo_features():
    """
    Demo function to compute and display features.
    """
    from aria.global_params import BENCHMARKS_PATH

    cnf_path = BENCHMARKS_PATH / "dimacs" / "parity_5.cnf"
    get_base_features(cnf_path)


if __name__ == "__main__":
    demo_features()
