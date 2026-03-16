"""
Wrappers for the frontends
- Constraint Horn Clause (CHC)
- SyGuS invariant track
- VMT: TBD
- Btor2: TBD
"""

from .mini_sygus_parser import SyGusInVParser
from .chc_parser import CHCParser
from .boogie2efmc import boogie_to_efmc
from .c2efmc import c_to_efmc


def parse_chc(filename: str, to_real_type: bool):
    """Parse CHC file"""
    with open(filename, "r", encoding="utf-8") as f:
        res = f.read()
        ss = CHCParser(res, to_real_type)
        return ss.get_transition_system()


def parse_sygus(filename: str, to_real_type: bool):
    """Parse SyGuS file"""
    with open(filename, "r", encoding="utf-8") as f:
        res = f.read()
        ss = SyGusInVParser(res, to_real_type)
        return ss.get_transition_system()


def parse_boogie(filename: str, to_real_type: bool = False):
    """Parse Boogie file"""
    # Note: to_real_type parameter is kept for consistency with other parsers
    # but Boogie parser doesn't use it currently
    return boogie_to_efmc(filename)


def parse_c(filename: str, to_real_type: bool = False):
    """Parse C file"""
    return c_to_efmc(filename)
