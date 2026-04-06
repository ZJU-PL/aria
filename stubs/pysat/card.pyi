from typing import Iterable, List, Optional

from pysat.formula import CNF, CNFPlus, IDPool


class NoSuchEncodingError(Exception): ...


class UnsupportedBound(Exception): ...


class EncType:
    pairwise: int
    seqcounter: int
    sortnetwrk: int
    cardnetwrk: int
    bitwise: int
    ladder: int
    totalizer: int
    mtotalizer: int
    kmtotalizer: int
    native: int


class CardEnc:
    @classmethod
    def atmost(
        cls,
        lits: Iterable[int],
        bound: int = ...,
        top_id: Optional[int] = ...,
        vpool: Optional[IDPool] = ...,
        encoding: int = ...,
    ) -> CNFPlus: ...
    @classmethod
    def atleast(
        cls,
        lits: Iterable[int],
        bound: int = ...,
        top_id: Optional[int] = ...,
        vpool: Optional[IDPool] = ...,
        encoding: int = ...,
    ) -> CNFPlus: ...
    @classmethod
    def equals(
        cls,
        lits: Iterable[int],
        bound: int = ...,
        top_id: Optional[int] = ...,
        vpool: Optional[IDPool] = ...,
        encoding: int = ...,
    ) -> CNFPlus: ...


class ITotalizer:
    lits: List[int]
    ubound: int
    top_id: int
    cnf: CNF
    rhs: List[int]
    nof_new: int

    def __init__(
        self, lits: Iterable[int] = ..., ubound: int = ..., top_id: Optional[int] = ...
    ) -> None: ...
    def new(
        self, lits: Iterable[int] = ..., ubound: int = ..., top_id: Optional[int] = ...
    ) -> None: ...
    def delete(self) -> None: ...
    def __enter__(self) -> ITotalizer: ...
    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None: ...
    def increase(self, ubound: int = ..., top_id: Optional[int] = ...) -> None: ...
    def extend(
        self,
        lits: Iterable[int] = ...,
        ubound: Optional[int] = ...,
        top_id: Optional[int] = ...,
    ) -> None: ...
    def merge_with(
        self, another: ITotalizer, ubound: Optional[int] = ..., top_id: Optional[int] = ...
    ) -> None: ...
