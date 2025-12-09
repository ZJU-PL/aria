# basic solver
import sys
from typing import Dict, Tuple

import z3  # type: ignore

from .core import CliCmd, add_bool_argument  # type : ignore
from .horndb import load_horn_db_from_file  # type: ignore


def parse_z3_arg(s: str) -> Tuple[str, object]:
    k, v = s.split("=")
    if v == "false" or v == "False":
        v = False
    elif v == "true" or v == "True":
        v = True
    elif v.isnumeric():
        v = int(v)
    return k, v


def parse_yaml_options(fname: str):
    import yaml

    with open(fname) as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)

        if "spacer_opts" in data:
            data = data["spacer_opts"]
        else:
            data = None

        return data
    return None


def chc_solve_with_fp(db, args, opts: Dict[str, object]):
    if args.fresh:
        ctx = z3.Context()
    else:
        ctx = db.get_ctx()

    fp = z3.Fixedpoint(ctx=ctx)
    db.mk_fixedpoint(fp=fp)

    if args.st:
        fp.set("print_statistics", True)
    if args.pp:
        fp.set("xform.slice", True)
        fp.set("xform.inline_linear", True)
        fp.set("xform.inline_eager", True)
    else:
        fp.set("xform.slice", False)
        fp.set("xform.inline_linear", False)
        fp.set("xform.inline_eager", False)

    if args.spctr is not None:
        fp.set("spacer.trace_file", args.spctr)

    for k, v in opts.items():
        fp.set(k, v)

    for q in db.get_queries():
        qf = q.mk_query()
        if qf.ctx != ctx:
            qf = qf.translate(ctx)
        res = fp.query(qf)
    return res


def chc_solve_with_cli(fname: str, args, opts: Dict[str, object]) -> None:
    cmd = [args.z3]

    if args.verbose > 0:
        cmd.append("-v:{}".format(args.verbose))
    for tr in args.tr:
        cmd.append("-tr:{}".format(args.tr))

    def bool2str(v):
        if v is True:
            return "true"
        elif v is False:
            return "false"
        return str(v)

    if not args.pp:
        opts["xform.slice"] = False
        opts["xform.inline_linear"] = False
        opts["xform.inline_eager"] = False

    if args.spctr is not None:
        opts["spacer.trace_file"] = args.spctr

    for k, v in opts.items():
        cmd.append("fp.{}={}".format(k, bool2str(v)))

    cmd.append(fname)
    # do not actually run, just print the command line
    print(" ".join(cmd))


class ChcSolveCmd(CliCmd):
    def __init__(self) -> None:
        super().__init__("chcsolve", "ChcSolve", allow_extra=True)

    def mk_arg_parser(self, ap):
        ap = super().mk_arg_parser(ap)
        ap.add_argument("in_file", metavar="FILE", help="Input file")

        ap.add_argument(
            "--verbose",
            "-v",
            metavar="LEVEL",
            help="Verbosity level",
            type=int,
            default=0,
        )
        add_bool_argument(
            ap, "fresh", default=False, help="Use fresh context for solving"
        )
        add_bool_argument(ap, "pp", default=False, help="pre-processing")
        add_bool_argument(ap, "st", default=True, help="print statistics")
        ap.add_argument(
            "--spctr", default=None, metavar="FILE", help="spacer trace file"
        )
        ap.add_argument("--tr", action="append", default=[])
        ap.add_argument(
            "-y", metavar="FILE", dest="yaml", help="Yaml configuration file"
        )
        ap.add_argument(
            "--solver",
            choices=["fp", "cli", "smt"],
            default="fp",
            help="Solver method to use",
        )
        ap.add_argument(
            "--z3", help="Path to z3 executable", default="z3", metavar="FILE"
        )
        return ap

    def run(self, args, extra):
        opts = dict()
        if args.yaml is not None:
            opts = parse_yaml_options(args.yaml)
        for a in extra:
            k, v = parse_z3_arg(a)
            opts[k] = v

        if args.solver == "fp":
            db = load_horn_db_from_file(args.in_file)
            z3.set_param(verbose=args.verbose)
            for tr_lvl in args.tr:
                z3.enable_trace(tr_lvl)
            res = chc_solve_with_fp(db, args, opts)
            print(res)
        elif args.solver == "cli":
            chc_solve_with_cli(args.in_file, args, opts)
        return 0


if __name__ == "__main__":
    cmd = ChcSolveCmd()
    sys.exit(cmd.main(sys.argv[1:]))
