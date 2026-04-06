from decimal import Decimal
from enum import Enum
from typing import Any, Iterable, Iterator, List, Optional, Sequence, TextIO, Tuple, Union

Literal = int
Clause = List[int]
Weight = Union[int, Decimal]
AtMost = Union[Tuple[List[int], int], Tuple[List[int], int, List[int]]]


class IDPool:
    top: int
    obj2id: Any
    id2obj: Any
    _occupied: List[Tuple[int, int]]

    def __init__(
        self,
        start_from: int = ...,
        occupied: Optional[Iterable[Iterable[int]]] = ...,
        with_neg: bool = ...,
    ) -> None: ...
    def restart(
        self,
        start_from: int = ...,
        occupied: Optional[Iterable[Iterable[int]]] = ...,
        with_neg: bool = ...,
    ) -> None: ...
    def id(self, obj: Any = ...) -> int: ...
    def obj(self, vid: int) -> Any: ...
    def occupy(self, start: int, stop: int) -> None: ...


class FormulaError(Exception): ...


class FormulaType(Enum):
    ATOM: int
    AND: int
    OR: int
    NEG: int
    IMPL: int
    EQ: int
    XOR: int
    ITE: int
    CNF: int


class Formula:
    name: Optional[int]
    clauses: List[Clause]
    encoded: List[Clause]
    type: FormulaType

    @staticmethod
    def set_context(context: Any = ...) -> None: ...
    @staticmethod
    def attach_vpool(vpool: IDPool, context: Any = ...) -> None: ...
    @staticmethod
    def export_vpool(active: bool = ..., context: Any = ...) -> IDPool: ...
    @staticmethod
    def cleanup(context: Any = ...) -> None: ...
    @staticmethod
    def formulas(lits: Iterable[int], atoms_only: bool = ...) -> List[Formula]: ...
    @staticmethod
    def literals(forms: Iterable[Formula]) -> List[int]: ...
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def __invert__(self) -> Formula: ...
    def __neg__(self) -> Formula: ...
    def __and__(self, other: Formula) -> Formula: ...
    def __iand__(self, other: Formula) -> Formula: ...
    def __or__(self, other: Formula) -> Formula: ...
    def __ior__(self, other: Formula) -> Formula: ...
    def __rshift__(self, other: Formula) -> Formula: ...
    def __irshift__(self, other: Formula) -> Formula: ...
    def __lshift__(self, other: Formula) -> Formula: ...
    def __ilshift__(self, other: Formula) -> Formula: ...
    def __matmul__(self, other: Formula) -> Formula: ...
    def __imatmul__(self, other: Formula) -> Formula: ...
    def __xor__(self, other: Formula) -> Formula: ...
    def __ixor__(self, other: Formula) -> Formula: ...
    def __iter__(self) -> Iterator[Clause]: ...
    def clausify(self) -> None: ...
    def simplified(self, assumptions: Iterable[Formula] = ...) -> Formula: ...
    def satisfied(self, model: Sequence[int]) -> bool: ...
    def atoms(self, constants: bool = ...) -> List[Any]: ...


class Atom(Formula):
    object: Any
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...


class And(Formula):
    subformulas: List[Formula]
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...


class Or(Formula):
    subformulas: List[Formula]
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...


class Neg(Formula):
    subformula: Formula
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...


class Implies(Formula):
    left: Formula
    right: Formula
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...


class Equals(Formula):
    subformulas: List[Formula]
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...


class XOr(Formula):
    subformulas: List[Formula]
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...


class ITE(Formula):
    cond: Formula
    cons1: Formula
    cons2: Formula
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...


class CNF(Formula):
    nv: int
    clauses: List[Clause]
    encoded: List[Clause]
    comments: List[str]
    vpool: IDPool
    inps: List[int]
    outs: List[int]
    auxvars: List[int]
    enclits: List[int]

    def __init__(
        self,
        from_file: Optional[str] = ...,
        from_fp: Optional[TextIO] = ...,
        from_string: Optional[str] = ...,
        from_clauses: Optional[Sequence[Sequence[int]]] = ...,
        from_aiger: Any = ...,
        comment_lead: Sequence[str] = ...,
        by_ref: bool = ...,
    ) -> None: ...
    def from_file(
        self,
        fname: str,
        comment_lead: Sequence[str] = ...,
        compressed_with: Optional[str] = ...,
    ) -> None: ...
    def from_fp(self, file_pointer: TextIO, comment_lead: Sequence[str] = ...) -> None: ...
    def from_string(self, string: str, comment_lead: Sequence[str] = ...) -> None: ...
    def from_clauses(self, clauses: Sequence[Sequence[int]], by_ref: bool = ...) -> None: ...
    def from_aiger(self, aig: Any, vpool: Optional[IDPool] = ...) -> None: ...
    def copy(self) -> CNF: ...
    def to_file(
        self,
        fname: str,
        comments: Optional[Sequence[str]] = ...,
        as_dnf: bool = ...,
        compress_with: Optional[str] = ...,
    ) -> None: ...
    def to_fp(
        self,
        file_pointer: TextIO,
        comments: Optional[Sequence[str]] = ...,
        as_dnf: bool = ...,
    ) -> None: ...
    def to_dimacs(self) -> str: ...
    def to_alien(
        self,
        file_pointer: TextIO,
        format: str = ...,
        comments: Optional[Sequence[str]] = ...,
    ) -> None: ...
    def append(self, clause: Iterable[int], update_vpool: bool = ...) -> None: ...
    def extend(self, clauses: Iterable[Iterable[int]]) -> None: ...
    def __iter__(self) -> Iterator[Clause]: ...
    def weighted(self) -> WCNF: ...
    def negate(self, topv: Optional[int] = ...) -> CNF: ...
    def simplified(self, assumptions: Iterable[Formula] = ...) -> Formula: ...


class WCNF:
    nv: int
    hard: List[Clause]
    soft: List[Clause]
    wght: List[Weight]
    topw: Weight
    comments: List[str]

    def __init__(
        self,
        from_file: Optional[str] = ...,
        from_fp: Optional[TextIO] = ...,
        from_string: Optional[str] = ...,
        comment_lead: Sequence[str] = ...,
    ) -> None: ...
    def from_file(
        self,
        fname: str,
        comment_lead: Sequence[str] = ...,
        compressed_with: Optional[str] = ...,
    ) -> None: ...
    def from_fp(self, file_pointer: TextIO, comment_lead: Sequence[str] = ...) -> None: ...
    def normalize_negatives(self, negatives: Iterable[Tuple[Clause, Weight]]) -> None: ...
    def from_string(self, string: str, comment_lead: Sequence[str] = ...) -> None: ...
    def copy(self) -> WCNF: ...
    def to_file(
        self,
        fname: str,
        comments: Optional[Sequence[str]] = ...,
        compress_with: Optional[str] = ...,
        format: str = ...,
    ) -> None: ...
    def to_fp(
        self,
        file_pointer: TextIO,
        comments: Optional[Sequence[str]] = ...,
        format: str = ...,
    ) -> None: ...
    def to_dimacs(self, format: str = ...) -> str: ...
    def to_alien(
        self,
        file_pointer: TextIO,
        format: str = ...,
        comments: Optional[Sequence[str]] = ...,
    ) -> None: ...
    def append(self, clause: Iterable[int], weight: Optional[Weight] = ...) -> None: ...
    def extend(
        self,
        clauses: Iterable[Iterable[int]],
        weights: Optional[Sequence[Weight]] = ...,
    ) -> None: ...
    def unweighted(self) -> CNF: ...


class CNFPlus:
    nv: int
    clauses: List[Clause]
    encoded: List[Clause]
    comments: List[str]
    atmosts: List[AtMost]

    def __init__(
        self,
        from_file: Optional[str] = ...,
        from_fp: Optional[TextIO] = ...,
        from_string: Optional[str] = ...,
        comment_lead: Sequence[str] = ...,
    ) -> None: ...
    def from_fp(self, file_pointer: TextIO, comment_lead: Sequence[str] = ...) -> None: ...
    def to_fp(self, file_pointer: TextIO, comments: Optional[Sequence[str]] = ...) -> None: ...
    def to_dimacs(self) -> str: ...
    def to_alien(
        self,
        file_pointer: TextIO,
        format: str = ...,
        comments: Optional[Sequence[str]] = ...,
    ) -> None: ...
    def append(
        self,
        clause: Union[Iterable[int], AtMost],
        is_atmost: bool = ...,
    ) -> None: ...
    def extend(self, formula: Union[CNFPlus, Iterable[Union[Iterable[int], AtMost]]]) -> None: ...
    def __iter__(self) -> Iterator[Union[Clause, AtMost]]: ...
    def weighted(self) -> WCNFPlus: ...
    def copy(self) -> CNFPlus: ...


class WCNFPlus:
    nv: int
    hard: List[Clause]
    soft: List[Clause]
    wght: List[Weight]
    topw: Weight
    comments: List[str]
    atms: List[AtMost]

    def __init__(
        self,
        from_file: Optional[str] = ...,
        from_fp: Optional[TextIO] = ...,
        from_string: Optional[str] = ...,
        comment_lead: Sequence[str] = ...,
    ) -> None: ...
    def from_fp(self, file_pointer: TextIO, comment_lead: Sequence[str] = ...) -> None: ...
    def to_fp(
        self,
        file_pointer: TextIO,
        comments: Optional[Sequence[str]] = ...,
        format: str = ...,
    ) -> None: ...
    def to_dimacs(self, format: str = ...) -> str: ...
    def to_alien(
        self,
        file_pointer: TextIO,
        format: str = ...,
        comments: Optional[Sequence[str]] = ...,
    ) -> None: ...
    def append(
        self,
        clause: Union[Iterable[int], AtMost],
        weight: Optional[Weight] = ...,
        is_atmost: bool = ...,
    ) -> None: ...
    def unweighted(self) -> CNFPlus: ...
    def copy(self) -> WCNFPlus: ...


PYSAT_FALSE: Formula
PYSAT_TRUE: Formula
aiger_present: bool
