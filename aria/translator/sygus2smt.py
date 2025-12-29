"""
This files takes a SyGuS instance in bit-vector theory and convert it to SMT instance
TODO: is this finished?
"""

from __future__ import print_function
import argparse
import re


def convert_to_smt(slfile, smtfile):
    """Convert a SyGuS file to SMT format."""
    with open(slfile, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    content = ''
    flag = 0
    for line in lines:
        if line.startswith("(synth-fun"):
            line = line.replace("(synth-fun", "(declare-fun")
            varname_pattern = r"\(([a-z]*[A-Z]*[0-9]*)\s"
            varname = re.findall(varname_pattern, line)
            for var in varname:
                if var != "":
                    line = line.replace("(" + var, "")

            line = line.replace("Bool)", "Bool")
            line = line.replace("))", ")")
            if "(BitVec " in line:
                line = line.replace("(BitVec", "(_ BitVec")
            content += line.strip("\n") + ")\n"
            flag = 1
            continue
        if flag == 1:
            if "declare" in line or "define" in line or "constraint" in line:
                flag = 0
        if flag == 0:
            if "(BitVec " in line:
                line = line.replace("(BitVec", "(_ BitVec")
            if "declare-var " in line:
                line = line.replace("declare-var", "declare-const")
            if "check-synth" in line:
                line = line.replace("check-synth", "check-sat")
            if "constraint" in line:
                line = line.replace("constraint", "assert")
            content += line
    with open(smtfile, "w", encoding='utf-8') as f:
        f.write(content)


def convert_sl_to_smt(file):
    """Convert a .sl file to .smt2 format."""
    filename = file.split(".sl")[0]
    convert_to_smt(file, filename + ".smt2")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, help="sl file", dest='file')
    args = parser.parse_args()
    convert_sl_to_smt(args.file)
