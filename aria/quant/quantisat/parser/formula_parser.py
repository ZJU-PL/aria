from warnings import warn

from .formula_m_parser import parse_file as parse_m_file
from .formula_smt_parser import parse_file as parse_smt_file


def parse_file(file_path):
    """
    Parse the formula from the given file.

    Parameters
    ----------
    file_path : str
        The path to the file to be parsed.

    Returns
    -------
    [Formula]
        The parsed formula.
    """
    if file_path.endswith('.m'):
        formula = parse_m_file(file_path)
    elif file_path.endswith('.smt2'):
        formula = parse_smt_file(file_path)
    else:
        warn(f'Unknown file extension: {file_path}')
        return None

    return formula
