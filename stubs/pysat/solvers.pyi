from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

Clause = Sequence[int]
Formula = Iterable[Iterable[int]]
Stats = Dict[str, int]
Parameters = Mapping[str, Union[int, float, bool]]


class NoSuchSolverError(Exception): ...


class SolverNames:
    cadical103: Tuple[str, ...]
    cadical153: Tuple[str, ...]
    cadical195: Tuple[str, ...]
    cryptosat: Tuple[str, ...]
    gluecard3: Tuple[str, ...]
    gluecard4: Tuple[str, ...]
    glucose3: Tuple[str, ...]
    glucose4: Tuple[str, ...]
    glucose42: Tuple[str, ...]
    kissat404: Tuple[str, ...]
    lingeling: Tuple[str, ...]
    maplechrono: Tuple[str, ...]
    maplecm: Tuple[str, ...]
    maplesat: Tuple[str, ...]
    mergesat3: Tuple[str, ...]
    minicard: Tuple[str, ...]
    minisat22: Tuple[str, ...]
    minisatgh: Tuple[str, ...]


class _SolverBase:
    status: Optional[bool]
    use_timer: bool
    call_time: float
    accu_time: float

    def __enter__(self) -> _SolverBase: ...
    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None: ...
    def delete(self) -> None: ...
    def start_mode(self, warm: bool = ...) -> None: ...
    def configure(self, parameters: Parameters) -> None: ...
    def activate_atmost(self) -> None: ...
    def connect_propagator(self, propagator: Any) -> None: ...
    def disconnect_propagator(self) -> None: ...
    def enable_propagator(self) -> None: ...
    def disable_propagator(self) -> None: ...
    def propagator_active(self) -> bool: ...
    def observe(self, var: int) -> None: ...
    def ignore(self, var: int) -> None: ...
    def reset_observed(self) -> None: ...
    def is_decision(self, lit: int) -> bool: ...
    def accum_stats(self) -> Stats: ...
    def solve(self, assumptions: Iterable[int] = ...) -> Optional[bool]: ...
    def solve_limited(
        self, assumptions: Iterable[int] = ..., expect_interrupt: bool = ...
    ) -> Optional[bool]: ...
    def conf_budget(self, budget: int = ...) -> None: ...
    def prop_budget(self, budget: int = ...) -> None: ...
    def dec_budget(self, budget: int) -> None: ...
    def interrupt(self) -> None: ...
    def clear_interrupt(self) -> None: ...
    def propagate(
        self, assumptions: Iterable[int] = ..., phase_saving: int = ...
    ) -> Tuple[bool, List[int]]: ...
    def set_phases(self, literals: Iterable[int] = ...) -> None: ...
    def get_status(self) -> Optional[bool]: ...
    def get_model(self) -> Optional[List[int]]: ...
    def get_core(self) -> Optional[List[int]]: ...
    def get_proof(self) -> Optional[List[str]]: ...
    def time(self) -> float: ...
    def time_accum(self) -> float: ...
    def nof_vars(self) -> int: ...
    def nof_clauses(self) -> int: ...
    def enum_models(self, assumptions: Iterable[int] = ...) -> Iterator[List[int]]: ...
    def add_clause(self, clause: Clause, no_return: bool = ...) -> Optional[bool]: ...
    def add_atmost(
        self,
        lits: Iterable[int],
        k: int,
        weights: Iterable[int] = ...,
        no_return: bool = ...,
    ) -> Optional[bool]: ...
    def add_xor_clause(self, lits: Iterable[int], value: bool = ...) -> None: ...
    def append_formula(self, formula: Iterable[Any], no_return: bool = ...) -> Optional[bool]: ...
    def supports_atmost(self) -> bool: ...


class Solver(_SolverBase):
    solver: Optional[_SolverBase]

    def __init__(
        self,
        name: str = ...,
        bootstrap_with: Optional[Iterable[Any]] = ...,
        use_timer: bool = ...,
        **kwargs: Any,
    ) -> None: ...
    def new(
        self,
        name: str = ...,
        bootstrap_with: Optional[Iterable[Any]] = ...,
        use_timer: bool = ...,
        **kwargs: Any,
    ) -> None: ...
    @staticmethod
    def _proof_bin2text(bytes_: bytearray) -> List[str]: ...


class Cadical103(_SolverBase):
    def __init__(
        self,
        bootstrap_with: Optional[Iterable[Any]] = ...,
        use_timer: bool = ...,
        incr: bool = ...,
        with_proof: bool = ...,
        warm_start: bool = ...,
    ) -> None: ...


class Cadical153(_SolverBase):
    def __init__(
        self,
        bootstrap_with: Optional[Iterable[Any]] = ...,
        use_timer: bool = ...,
        incr: bool = ...,
        with_proof: bool = ...,
        warm_start: bool = ...,
    ) -> None: ...
    def process(self, **kwargs: Any) -> Any: ...
    def restore(self, model: Sequence[int]) -> Any: ...


class Cadical195(_SolverBase):
    def __init__(
        self,
        bootstrap_with: Optional[Iterable[Any]] = ...,
        use_timer: bool = ...,
        incr: bool = ...,
        with_proof: bool = ...,
        warm_start: bool = ...,
        native_card: bool = ...,
    ) -> None: ...
    def process(self, **kwargs: Any) -> Any: ...
    def restore(self, model: Sequence[int]) -> Any: ...


class Gluecard3(_SolverBase):
    def __init__(
        self,
        bootstrap_with: Optional[Iterable[Any]] = ...,
        use_timer: bool = ...,
        incr: bool = ...,
        with_proof: bool = ...,
        warm_start: bool = ...,
    ) -> None: ...


class Gluecard4(_SolverBase):
    def __init__(
        self,
        bootstrap_with: Optional[Iterable[Any]] = ...,
        use_timer: bool = ...,
        incr: bool = ...,
        with_proof: bool = ...,
        warm_start: bool = ...,
    ) -> None: ...


class Glucose3(_SolverBase):
    def __init__(
        self,
        bootstrap_with: Optional[Iterable[Any]] = ...,
        use_timer: bool = ...,
        incr: bool = ...,
        with_proof: bool = ...,
        warm_start: bool = ...,
    ) -> None: ...


class Glucose4(_SolverBase):
    def __init__(
        self,
        bootstrap_with: Optional[Iterable[Any]] = ...,
        use_timer: bool = ...,
        incr: bool = ...,
        with_proof: bool = ...,
        warm_start: bool = ...,
    ) -> None: ...


class Glucose42(_SolverBase):
    def __init__(
        self,
        bootstrap_with: Optional[Iterable[Any]] = ...,
        use_timer: bool = ...,
        incr: bool = ...,
        with_proof: bool = ...,
        warm_start: bool = ...,
    ) -> None: ...
    def set_rnd_seed(self, seed: int) -> None: ...
    def set_rnd_freq(self, freq: float) -> None: ...
    def set_rnd_pol(self, to_enable: bool) -> None: ...
    def set_rnd_init_act(self, to_enable: bool) -> None: ...
    def set_rnd_first_descent(self, to_enable: bool) -> None: ...


class Lingeling(_SolverBase):
    def __init__(
        self,
        bootstrap_with: Optional[Iterable[Any]] = ...,
        use_timer: bool = ...,
        incr: bool = ...,
        with_proof: bool = ...,
        warm_start: bool = ...,
    ) -> None: ...


class MapleChrono(_SolverBase):
    def __init__(
        self,
        bootstrap_with: Optional[Iterable[Any]] = ...,
        use_timer: bool = ...,
        incr: bool = ...,
        with_proof: bool = ...,
        warm_start: bool = ...,
    ) -> None: ...


class MapleCM(_SolverBase):
    def __init__(
        self,
        bootstrap_with: Optional[Iterable[Any]] = ...,
        use_timer: bool = ...,
        incr: bool = ...,
        with_proof: bool = ...,
        warm_start: bool = ...,
    ) -> None: ...


class Maplesat(_SolverBase):
    def __init__(
        self,
        bootstrap_with: Optional[Iterable[Any]] = ...,
        use_timer: bool = ...,
        incr: bool = ...,
        with_proof: bool = ...,
        warm_start: bool = ...,
    ) -> None: ...


class Mergesat3(_SolverBase):
    def __init__(
        self,
        bootstrap_with: Optional[Iterable[Any]] = ...,
        use_timer: bool = ...,
    ) -> None: ...


class Minicard(_SolverBase):
    def __init__(
        self,
        bootstrap_with: Optional[Iterable[Any]] = ...,
        use_timer: bool = ...,
        incr: bool = ...,
        with_proof: bool = ...,
        warm_start: bool = ...,
    ) -> None: ...


class Minisat22(_SolverBase):
    def __init__(
        self,
        bootstrap_with: Optional[Iterable[Any]] = ...,
        use_timer: bool = ...,
        incr: bool = ...,
        with_proof: bool = ...,
        warm_start: bool = ...,
    ) -> None: ...


class MinisatGH(_SolverBase):
    def __init__(
        self,
        bootstrap_with: Optional[Iterable[Any]] = ...,
        use_timer: bool = ...,
        incr: bool = ...,
        with_proof: bool = ...,
        warm_start: bool = ...,
    ) -> None: ...


class CryptoMinisat(_SolverBase):
    def __init__(
        self,
        bootstrap_with: Optional[Iterable[Any]] = ...,
        use_timer: bool = ...,
        incr: bool = ...,
        with_proof: bool = ...,
        warm_start: bool = ...,
    ) -> None: ...
    def time_budget(self, budget: int) -> None: ...


class Kissat404(_SolverBase):
    def __init__(
        self,
        bootstrap_with: Optional[Iterable[Any]] = ...,
        use_timer: bool = ...,
        incr: bool = ...,
        with_proof: bool = ...,
        warm_start: bool = ...,
    ) -> None: ...


BooleanEngine: Any
cms_present: bool
