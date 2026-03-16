"""Module for interacting with CPAchecker verification tool."""

import os
import re
from os.path import dirname, abspath, relpath
from subprocess import call, check_output, STDOUT
from tempfile import NamedTemporaryFile

try:
    from pydot import graph_from_dot_file
except ImportError:
    graph_from_dot_file = None

from z3 import parse_smt2_string

from aria.efmc.verifytools.common.util import unique

MYDIR = dirname(abspath(relpath(__file__)))

CPA_PATH = (
    MYDIR + "/../../../env/third_party/cpa_checker_1.4/CPAchecker-1.4-svcomp16c-unix/"
)


def find_loop_header_label(dot_file):
    """Find the loop header label from a dot file."""
    g = unique(graph_from_dot_file(dot_file))
    nodes = g.get_nodes()
    shape_list = [(n.get_shape(), n.get_label()) for n in nodes]
    double_circles = [t for t in shape_list if t[0] == '"doublecircle"']
    # The loop header should be the unique node with a double circle shape
    # Its label will be a string of the form '"N[0-9]*\\n[0-9]"'. We care
    # just about the first [0-9]* part.
    return unique(double_circles)[1].split("\\n")[0][2:]


def parse_abstraction_file(fname):
    """Parse an abstraction file and extract invariants."""
    with open(fname, encoding="utf-8") as f:
        content = f.read()
    lines = filter(lambda x: x != "", map(lambda x: x.strip(), content.split("\n")))
    decls = []
    invs = {}
    label_re = re.compile(r"^(?P<n1>[0-9]*) \((?P<n2>[0-9,]*)\) \@(?P<n3>[0-9]*):$")
    var_re = re.compile(r"\|[a-zA-Z0-9]*::([a-zA-Z0-9_]*)\|")
    cur_lbl = None
    for l in lines:
        if l.startswith("(declare-fun"):
            decls.append(var_re.sub(r"\1", l))
        elif label_re.match(l):
            cur_lbl = label_re.match(l).groupdict()["n3"]
        else:
            assert cur_lbl is not None
            full_str = "\n".join(decls + [var_re.sub(r"\1", l)])
            invs[cur_lbl] = invs.get(cur_lbl, []) + [parse_smt2_string(full_str)]
    return invs


def parse_invariants_file(fname):
    """Parse an invariants file and extract the invariant."""
    with open(fname, encoding="utf-8") as f:
        content = f.read()
    lines = filter(lambda x: x != "", map(lambda x: x.strip(), content.split("\n")))
    label_re = re.compile(r"^[^ ]* [^ :]*:$")
    label_lines = [l for l in lines if label_re.match(l)]
    assert len(label_lines) == 1  # Single loop header so single invariant

    lines = [l for l in lines if not label_re.match(l)]
    var_re = re.compile(r"\|[a-zA-Z0-9]*::([a-zA-Z0-9_]*)\|")

    full_str = "\n".join([var_re.sub(r"\1", l) for l in lines])
    return parse_smt2_string(full_str)


def get_loop_invariants(output_dir):
    """Get loop invariants from CPAchecker output directory."""
    loop_header = find_loop_header_label(output_dir + "/cfa.dot")
    invs = parse_abstraction_file(output_dir + "/abstractions.txt")
    inv = parse_invariants_file(output_dir + "/invariantPrecs.txt")
    return invs[loop_header] + [inv]


def convert_cpp_file_for_cpa_checker(cpp_file, out_file):
    """Convert a C++ file for CPAchecker processing."""
    cpp_args = [
        "cpp",
        "-include",
        MYDIR + "/dummy.h",
        "-D_Static_assert=__tmp_assert",
        "-D_static_assert=__tmp_assert",
        "-Dstatic_assert=__tmp_assert",
        "-D__VERIFIER_assert=__tmp_assert",
        "-D__VERIFIER_assume(a)=assume(a)",
        "-Dassume(a)=if(!(a)) exit(0)",
        cpp_file,
        out_file,
    ]

    call(cpp_args)


def run_cpa_checker(
    inp_file, timelimit=100, config="predicateAnalysis-ImpactRefiner-ABEl.properties"
):
    """Run CPAchecker on an input file and extract invariants."""
    nondet_functions = [
        "unknown1",
        "unknown2",
        "unknown3",
        "unknown4",
        "unknown5",
        "random",
        "__VERIFIER_nondet_int",
        "__VERIFIER_nondet_uint",
    ]
    args = [
        CPA_PATH + "scripts/cpa.sh",
        "-config",
        CPA_PATH + "config/" + config,
        "-timelimit",
        str(timelimit),
        "-setprop",
        "cpa.predicate.nondetFunctions=" + ",".join(nondet_functions),
        inp_file,
    ]
    raw = check_output(args, stderr=STDOUT)
    lines = raw.split("\n")
    lines = [
        x
        for x in lines
        if not (
            x.startswith("Running CPAchecker with")
            or x.startswith("Using the following resource")
            or x.startswith("CPAchecker 1.4-svcomp16c (OpenJDK")
            or x.startswith("Using predicate analysis with")
            or x.startswith("Using refinement for predicate analysis")
            or x.startswith("Starting analysis ...")
            or x.startswith("Stopping analysis ...")
            or x.startswith("More details about the verification run")
            or len(x.strip()) == 0
        )
    ]
    verified = len([x for x in lines if "Verification result: TRUE." in x]) > 0

    header_label = find_loop_header_label("output/cfa.dot")
    invs = get_loop_invariants("output/")
    return (verified, header_label, invs, "\n".join(lines))


# Backward compatibility aliases
findLoopHeaderLabel = find_loop_header_label
parseAbstractionFile = parse_abstraction_file
parseInvariantsFile = parse_invariants_file
getLoopInvariants = get_loop_invariants
convertCppFileForCPAChecker = convert_cpp_file_for_cpa_checker
runCPAChecker = run_cpa_checker
