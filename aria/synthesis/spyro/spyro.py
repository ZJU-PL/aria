"""Main entry point for Spyro synthesis."""

import argparse
import sys

from aria.synthesis.spyro.property_synthesizer import PropertySynthesizer


def main():
    """Run Spyro property synthesizer."""
    parser = argparse.ArgumentParser()
    parser.add_argument('infiles', nargs='+', type=argparse.FileType('r'))
    parser.add_argument('outfile', nargs='?', type=argparse.FileType('w'), default=sys.stdout)
    parser.add_argument('--write-log', dest='write_log', action='store_true', default=False)
    parser.add_argument('--verbose', '-v', dest='verbose', action='store_true', default=False)
    parser.add_argument('--timeout', dest='timeout', type=int, nargs='?', default=300)
    parser.add_argument('--inline-bnd', dest='inline_bnd', type=int, nargs='?', default=5)
    parser.add_argument('--num-atom-max', dest='num_atom_max', type=int, nargs='?', default=3)
    parser.add_argument('--disable-min', dest='disable_min', action='store_true', default=False)
    parser.add_argument('--keep-neg-may', dest='keep_neg_may', action='store_true', default=False)
    parser.add_argument('--slv-seed', dest='slv_seed', type=int, nargs='?', default=0)

    args = parser.parse_args(sys.argv[1:])
    infiles = args.infiles
    outfile = args.outfile
    v = args.verbose
    write_log = args.write_log
    timeout = args.timeout
    inline_bnd = args.inline_bnd
    num_atom_max = args.num_atom_max
    disable_min = args.disable_min
    keep_neg_may = args.keep_neg_may
    slv_seed = args.slv_seed

    phi_list, fun_list, _ = PropertySynthesizer(
        infiles, outfile, v, write_log,
        timeout, inline_bnd, slv_seed,
        num_atom_max, disable_min, keep_neg_may).run()

    for n in range(len(phi_list)):
        phi = phi_list[n]
        funs = fun_list[n]

        outfile.write(f"Property {n}\n\n")
        outfile.write(str(phi) + "\n")
        for function_name, code in funs:
            outfile.write(function_name + "\n")
            outfile.write(code + "\n")
        outfile.write("\n\n")

    for f in infiles:
        f.close()
    outfile.close()


if __name__ == "__main__":
    main()
