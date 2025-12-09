import argparse
import os
import os.path
from typing import Any, List, Optional


def str2bool(v: "bool | str") -> bool:
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def add_bool_argument(
    parser: argparse.ArgumentParser,
    name: str,
    default: bool = False,
    help: Optional[str] = None,
    dest: Optional[str] = None,
    **kwargs: Any,
) -> None:
    """
    Add boolean option that can be turned on and off
    """
    import argparse
    dest_name = dest if dest is not None else name
    mutex_group = parser.add_mutually_exclusive_group(required=False)
    mutex_group.add_argument('--' + name, dest=dest_name, type=str2bool,
                             nargs='?', const=True, help=help,
                             metavar='BOOL', **kwargs)
    mutex_group.add_argument('--no-' + name, dest=dest_name,
                             type=lambda v: not (str2bool(v)),
                             nargs='?', const=False,
                             help=argparse.SUPPRESS, **kwargs)
    parser.set_defaults(dest_name=default)


def add_help_arg(ap: argparse.ArgumentParser) -> None:
    ap.add_argument('-h', '--help', action='help',
                    help='Print this message and exit')


def add_in_args(ap: argparse.ArgumentParser) -> argparse.ArgumentParser:
    ap.add_argument('in_files', metavar='FILE', help='Input file', nargs='+')
    return ap


def add_in_out_args(ap: argparse.ArgumentParser) -> argparse.ArgumentParser:
    add_in_args(ap)
    ap.add_argument('-o', dest='out_file',
                    metavar='FILE', help='Output file name', default=None)
    return ap


def add_tmp_dir_args(ap: argparse.ArgumentParser) -> argparse.ArgumentParser:
    ap.add_argument('--save-temps', '--keep-temps',
                    dest="save_temps",
                    help="Do not delete temporary files",
                    action="store_true", default=False)
    ap.add_argument('--temp-dir', dest='temp_dir', metavar='DIR',
                    help="Temporary directory", default=None)
    return ap


class CliCmd(object):
    def __init__(self, name: str = '', help: str = '', allow_extra: bool = False) -> None:
        self.name: str = name
        self.help: str = help
        self.allow_extra: bool = allow_extra

    def mk_arg_parser(self, argp: argparse.ArgumentParser) -> argparse.ArgumentParser:
        add_help_arg(argp)
        return argp

    def run(self, args: Optional[argparse.Namespace] = None, extra: List[str] = []) -> int:
        return 0

    def name_out_file(
        self,
        in_files: List[str],
        args: Optional[argparse.Namespace] = None,
        work_dir: Optional[str] = None,
    ) -> str:
        out_file = 'out'
        if work_dir is not None:
            out_file = os.path.join(work_dir, out_file)
        return out_file

    def main(self, argv: List[str]) -> int:
        import argparse
        ap = argparse.ArgumentParser(prog=self.name,
                                     description=self.help,
                                     add_help=False)
        ap = self.mk_arg_parser(ap)

        if self.allow_extra:
            args, extra = ap.parse_known_args(argv)
        else:
            args = ap.parse_args(argv)
            extra = []
        return self.run(args, extra)
