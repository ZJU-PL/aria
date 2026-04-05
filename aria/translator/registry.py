from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class TranslationSpec:
    input_format: str
    output_format: str
    handler: Callable[[str, str], None]
    description: str
    output_extension: str


def _translate_dimacs_to_smt2(input_path: str, output_path: str) -> None:
    from aria.translator.dimacs2smt import (  # pylint: disable=import-outside-toplevel
        convert_dimacs_to_smt2,
    )

    convert_dimacs_to_smt2(input_path, output_path)


def _translate_dimacs_to_lp(input_path: str, output_path: str) -> None:
    from aria.translator.cnf2lp import cnf2lp  # pylint: disable=import-outside-toplevel

    cnf2lp(input_path, output_path)


def _translate_qdimacs_to_smt2(input_path: str, output_path: str) -> None:
    from aria.translator.qbf2smt import (  # pylint: disable=import-outside-toplevel
        convert_qdimacs_to_smt2,
    )

    convert_qdimacs_to_smt2(input_path, output_path)


def _translate_sygus_to_smt2(input_path: str, output_path: str) -> None:
    from aria.translator.sygus2smt import (  # pylint: disable=import-outside-toplevel
        convert_to_smt,
    )

    convert_to_smt(input_path, output_path)


def _translate_qcir_to_smt2(input_path: str, output_path: str) -> None:
    from aria.translator.qcir2smt import (  # pylint: disable=import-outside-toplevel
        convert_qcir_to_smt2,
    )

    convert_qcir_to_smt2(input_path, output_path)


def _translate_opb_to_smt2(input_path: str, output_path: str) -> None:
    from aria.translator.opb2smt import (  # pylint: disable=import-outside-toplevel
        convert_opb_to_smt2,
    )

    convert_opb_to_smt2(input_path, output_path)


def _translate_wcnf_to_smt2(input_path: str, output_path: str) -> None:
    from aria.translator.wcnf2smt import (  # pylint: disable=import-outside-toplevel
        convert_wcnf_to_smt2,
    )

    convert_wcnf_to_smt2(input_path, output_path)


def _translate_smtlib2_to_dimacs(input_path: str, output_path: str) -> None:
    from aria.translator.smt2dimacs import (  # pylint: disable=import-outside-toplevel
        convert_smt2_to_dimacs,
    )

    convert_smt2_to_dimacs(input_path, output_path)


def _translate_smtlib2_to_sympy(input_path: str, output_path: str) -> None:
    from aria.translator.smt2sympy import (  # pylint: disable=import-outside-toplevel
        smtlib_to_sympy_constraint,
    )

    with open(input_path, "r", encoding="utf-8") as input_file:
        content = input_file.read()

    expr = smtlib_to_sympy_constraint(content)

    with open(output_path, "w", encoding="utf-8") as output_file:
        output_file.write(f"{expr}\n")


_TRANSLATION_SPECS: List[TranslationSpec] = [
    TranslationSpec(
        input_format="dimacs",
        output_format="smtlib2",
        handler=_translate_dimacs_to_smt2,
        description="DIMACS CNF -> SMT-LIB2",
        output_extension=".smt2",
    ),
    TranslationSpec(
        input_format="dimacs",
        output_format="lp",
        handler=_translate_dimacs_to_lp,
        description="DIMACS CNF -> LP",
        output_extension=".lp",
    ),
    TranslationSpec(
        input_format="qdimacs",
        output_format="smtlib2",
        handler=_translate_qdimacs_to_smt2,
        description="QDIMACS -> SMT-LIB2",
        output_extension=".smt2",
    ),
    TranslationSpec(
        input_format="sygus",
        output_format="smtlib2",
        handler=_translate_sygus_to_smt2,
        description="SyGuS -> SMT-LIB2",
        output_extension=".smt2",
    ),
    TranslationSpec(
        input_format="qcir",
        output_format="smtlib2",
        handler=_translate_qcir_to_smt2,
        description="QCIR -> SMT-LIB2",
        output_extension=".smt2",
    ),
    TranslationSpec(
        input_format="opb",
        output_format="smtlib2",
        handler=_translate_opb_to_smt2,
        description="OPB -> SMT-LIB2",
        output_extension=".smt2",
    ),
    TranslationSpec(
        input_format="wcnf",
        output_format="smtlib2",
        handler=_translate_wcnf_to_smt2,
        description="WCNF -> SMT-LIB2",
        output_extension=".smt2",
    ),
    TranslationSpec(
        input_format="smtlib2",
        output_format="dimacs",
        handler=_translate_smtlib2_to_dimacs,
        description="SMT-LIB2 -> DIMACS",
        output_extension=".cnf",
    ),
]

if find_spec("pysmt") is not None:
    _TRANSLATION_SPECS.append(
        TranslationSpec(
            input_format="smtlib2",
            output_format="sympy",
            handler=_translate_smtlib2_to_sympy,
            description="SMT-LIB2 -> SymPy",
            output_extension=".sympy",
        )
    )

TRANSLATION_SPECS: Tuple[TranslationSpec, ...] = tuple(_TRANSLATION_SPECS)


TRANSLATION_MAP: Dict[Tuple[str, str], TranslationSpec] = {
    (spec.input_format, spec.output_format): spec for spec in TRANSLATION_SPECS
}

FORMAT_EXTENSIONS = {
    ".cnf": "dimacs",
    ".qdimacs": "qdimacs",
    ".qcir": "qcir",
    ".fzn": "fzn",
    ".smt2": "smtlib2",
    ".sy": "sygus",
    ".sl": "sygus",
    ".opb": "opb",
    ".wcnf": "wcnf",
    ".lp": "lp",
    ".dl": "datalog",
    ".sympy": "sympy",
}


def detect_format(filename: str) -> Optional[str]:
    return FORMAT_EXTENSIONS.get(Path(filename).suffix.lower())


def get_supported_translations() -> List[Tuple[str, str]]:
    return [(spec.input_format, spec.output_format) for spec in TRANSLATION_SPECS]


def get_supported_translation_formats() -> List[str]:
    formats = set()
    for spec in TRANSLATION_SPECS:
        formats.add(spec.input_format)
        formats.add(spec.output_format)
    return sorted(formats)


def get_output_extension(fmt: str) -> str:
    for spec in TRANSLATION_SPECS:
        if spec.output_format == fmt:
            return spec.output_extension
    return ".txt"


def iter_translation_specs() -> Iterable[TranslationSpec]:
    return TRANSLATION_SPECS


def translate_file(
    input_format: str, output_format: str, input_path: str, output_path: str
) -> None:
    pair = (input_format.lower(), output_format.lower())
    spec = TRANSLATION_MAP.get(pair)
    if spec is None:
        supported = ", ".join(
            f"{src} -> {dst}" for src, dst in get_supported_translations()
        )
        raise ValueError(
            f"No translator for {input_format} -> {output_format}. Supported: {supported}"
        )

    spec.handler(input_path, output_path)
