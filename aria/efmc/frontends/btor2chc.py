"""
Converting Btor2 to CHC

from https://github.com/stepwise-alan/btor2chc
"""

import sys
from abc import ABC, abstractmethod
from typing import Optional, List, TextIO, Union, Dict, Tuple, Callable, Any

import z3  # type: ignore


def to_z3_bitvec(z3_expr: Union[z3.BitVecRef, z3.BoolRef]) -> z3.BitVecRef:
    if isinstance(z3_expr, z3.BitVecRef):
        return z3_expr
    if z3.is_true(z3_expr):
        return z3.BitVecVal(1, 1, z3_expr.ctx)
    if z3.is_false(z3_expr):
        return z3.BitVecVal(0, 1, z3_expr.ctx)
    return z3.If(
        z3_expr, z3.BitVecVal(1, 1, z3_expr.ctx), z3.BitVecVal(0, 1, z3_expr.ctx)
    )


def to_z3_bool(z3_expr: Union[z3.BitVecRef, z3.BoolRef]) -> z3.BoolRef:
    if isinstance(z3_expr, z3.BoolRef):
        return z3_expr
    if isinstance(z3_expr, z3.BitVecNumRef):
        return z3.BoolVal(z3_expr, z3_expr.ctx)
    return z3_expr == z3.BitVecVal(1, 1, z3_expr.ctx)


def try_bitvec_to_bool(z3_expr: z3.ExprRef) -> z3.ExprRef:
    if isinstance(z3_expr, z3.BitVecRef) and z3_expr.size() == 1:
        return to_z3_bool(z3_expr)
    return z3_expr


def to_z3_eq(z3_expr1: z3.ExprRef, z3_expr2: z3.ExprRef) -> z3.BoolRef:
    if isinstance(z3_expr1, z3.BoolRef) or isinstance(z3_expr2, z3.BoolRef):
        return to_z3_bool(z3_expr1) == to_z3_bool(z3_expr2)
    return z3_expr1 == z3_expr2


# noinspection PyProtectedMember
def call_z3_mk_bv(
    z3_mk_bv: Callable[[Any, Any, Any], Any],
    z3_bitvec1: z3.BitVecRef,
    z3_bitvec2: z3.BitVecRef,
) -> z3.BitVecRef:
    from z3.z3 import _coerce_exprs  # type: ignore

    a, b = _coerce_exprs(z3_bitvec1, z3_bitvec2)
    return z3.BitVecRef(
        z3_mk_bv(z3_bitvec1.ctx_ref(), a.as_ast(), b.as_ast()), z3_bitvec1.ctx
    )


class Z3ExprMemo:
    ctx: z3.Context
    z3_bitvec_or_bool_m: Dict[int, Union[z3.BitVecRef, z3.BoolRef]]
    z3_array_m: Dict[int, z3.ArrayRef]

    def __init__(self, ctx: z3.Context = z3.Context()):
        self.ctx = ctx
        self.z3_bitvec_or_bool_m = {}
        self.z3_array_m = {}


class Node(ABC):
    symbol: str

    def __init__(self, symbol: str = ""):
        self.symbol = symbol


class Sort(Node):
    sid: int

    def __init__(self, sid: int, symbol: str = ""):
        super().__init__(symbol)
        self.sid = sid

    @abstractmethod
    def to_z3_sort(self, ctx: z3.Context = z3.main_ctx()) -> z3.SortRef:
        raise NotImplementedError


class BitvecSort(Sort):
    width: int

    def __init__(self, sid: int, width: int, symbol: str = ""):
        super().__init__(sid, symbol)
        self.width = width

    def to_z3_sort(
        self, ctx: z3.Context = z3.main_ctx()
    ) -> Union[z3.BitVecSortRef, z3.BoolSortRef]:
        if self.width == 1:
            return z3.BoolSort(ctx)
        return z3.BitVecSort(self.width, ctx)


class ArraySort(Sort):
    index_sort: Sort
    element_sort: Sort

    def __init__(
        self, sid: int, index_sort: Sort, element_sort: Sort, symbol: str = ""
    ):
        super().__init__(sid, symbol)
        self.index_sort = index_sort
        self.element_sort = element_sort

    def to_z3_sort(self, ctx: z3.Context = z3.main_ctx()) -> z3.ArraySortRef:
        return z3.ArraySort(
            self.index_sort.to_z3_sort(ctx), self.element_sort.to_z3_sort(ctx)
        )


class Expr(Node):
    nid: int
    sort: Sort

    def __init__(self, nid: int, sort: Sort, symbol: str = ""):
        super().__init__(symbol)
        self.nid = nid
        self.sort = sort

    @abstractmethod
    def to_z3_expr(self, m: Z3ExprMemo) -> z3.ExprRef:
        raise NotImplementedError


class Bitvec(Expr, ABC):
    sort: BitvecSort

    def __init__(self, nid: int, sort: BitvecSort, symbol: str = ""):
        super().__init__(nid, sort, symbol)

    def to_z3_expr(self, m: Z3ExprMemo) -> Union[z3.BitVecRef, z3.BoolRef]:
        if self.nid in m.z3_bitvec_or_bool_m:
            return m.z3_bitvec_or_bool_m[self.nid]
        z3_expr: Union[z3.BitVecRef, z3.BoolRef] = self.to_new_z3_expr(m)
        m.z3_bitvec_or_bool_m[self.nid] = z3_expr
        return z3_expr

    @abstractmethod
    def to_new_z3_expr(self, m: Z3ExprMemo) -> Union[z3.BitVecRef, z3.BoolRef]:
        raise NotImplementedError

    def to_z3_bitvec(self, m: Z3ExprMemo) -> z3.BitVecRef:
        return to_z3_bitvec(self.to_z3_expr(m))

    def to_z3_bool(self, m: Z3ExprMemo) -> z3.BoolRef:
        return to_z3_bool(self.to_z3_expr(m))


class Array(Expr, ABC):
    sort: ArraySort

    def __init__(self, nid: int, sort: ArraySort, symbol: str = ""):
        super().__init__(nid, sort, symbol)

    def to_z3_expr(self, m: Z3ExprMemo) -> z3.ArrayRef:
        if self.nid in m.z3_array_m:
            return m.z3_array_m[self.nid]
        z3_expr: z3.ArrayRef = self.to_new_z3_expr(m)
        m.z3_array_m[self.nid] = z3_expr
        return z3_expr

    @abstractmethod
    def to_new_z3_expr(self, m: Z3ExprMemo) -> z3.ArrayRef:
        raise NotImplementedError


class Minus(Bitvec):
    bitvec: Bitvec

    def __init__(self, bitvec: Bitvec):
        super().__init__(-bitvec.nid, bitvec.sort)
        self.bitvec = bitvec

    def to_new_z3_expr(self, m: Z3ExprMemo) -> Union[z3.BitVecRef, z3.BoolRef]:
        z3_expr: Union[z3.BitVecRef, z3.BoolRef] = self.bitvec.to_z3_expr(m)
        if isinstance(z3_expr, z3.BitVecRef):
            return ~z3_expr
        return z3.Not(z3_expr)


class BitvecInput(Bitvec):
    def to_new_z3_expr(self, m: Z3ExprMemo) -> Union[z3.BitVecRef, z3.BoolRef]:
        raise ValueError


class ArrayInput(Array):
    def to_new_z3_expr(self, m: Z3ExprMemo) -> z3.ArrayRef:
        raise ValueError


class Constant(Bitvec):
    value: int

    def __init__(self, nid: int, sort: BitvecSort, value: int, symbol: str = ""):
        super().__init__(nid, sort, symbol)
        self.value = value

    def to_new_z3_expr(self, m: Z3ExprMemo) -> Union[z3.BitVecRef, z3.BoolRef]:
        if self.sort.width == 1:
            return z3.BoolVal(self.value, m.ctx)
        return z3.BitVecVal(self.value, self.sort.width, m.ctx)


class One(Constant):
    def __init__(self, nid: int, sort: BitvecSort, symbol: str = ""):
        super().__init__(nid, sort, 1, symbol)


class Ones(Constant):
    def __init__(self, nid: int, sort: BitvecSort, symbol: str = ""):
        super().__init__(nid, sort, -1, symbol)


class Zero(Constant):
    def __init__(self, nid: int, sort: BitvecSort, symbol: str = ""):
        super().__init__(nid, sort, 0, symbol)


class Const(Constant):
    def __init__(self, nid: int, sort: BitvecSort, bin_str: str, symbol: str = ""):
        super().__init__(nid, sort, int(bin_str, 2), symbol)


class Constd(Constant):
    def __init__(self, nid: int, sort: BitvecSort, dec_str: str, symbol: str = ""):
        super().__init__(nid, sort, int(dec_str, 10), symbol)


class Consth(Constant):
    def __init__(self, nid: int, sort: BitvecSort, hex_str: str, symbol: str = ""):
        super().__init__(nid, sort, int(hex_str, 16), symbol)


class State:
    init: Optional[Expr]
    next: Optional[Expr]

    def __init__(self):
        self.init = None
        self.next = None


class BitvecState(Bitvec, State):
    init: Optional[Bitvec]
    next: Optional[Bitvec]

    def __init__(self, nid: int, sort: BitvecSort, symbol: str = ""):
        Bitvec.__init__(self, nid, sort, symbol)
        State.__init__(self)

    def to_new_z3_expr(self, m: Z3ExprMemo) -> Union[z3.BitVecRef, z3.BoolRef]:
        raise ValueError


class ArrayState(Array, State):
    next: Optional[Array]

    def __init__(self, nid: int, sort: ArraySort, symbol: str = ""):
        Array.__init__(self, nid, sort, symbol)
        State.__init__(self)

    def to_new_z3_expr(self, m: Z3ExprMemo) -> z3.ArrayRef:
        raise ValueError


class Ext(Bitvec, ABC):
    bitvec: Bitvec
    w: int

    def __init__(
        self, nid: int, sort: BitvecSort, bitvec: Bitvec, w: int, symbol: str = ""
    ):
        super().__init__(nid, sort, symbol)
        self.bitvec = bitvec
        self.w = w


class Sext(Ext):
    def to_new_z3_expr(self, m: Z3ExprMemo) -> z3.BitVecRef:
        return z3.SignExt(self.w, self.bitvec.to_z3_bitvec(m))


class Uext(Ext):
    def to_new_z3_expr(self, m: Z3ExprMemo) -> z3.BitVecRef:
        return z3.ZeroExt(self.w, self.bitvec.to_z3_bitvec(m))


class Slice(Bitvec):
    bitvec: Bitvec
    upper: int
    lower: int

    def __init__(
        self,
        nid: int,
        sort: BitvecSort,
        bitvec: Bitvec,
        upper: int,
        lower: int,
        symbol: str = "",
    ):
        super().__init__(nid, sort, symbol)
        self.bitvec = bitvec
        self.upper = upper
        self.lower = lower

    def to_new_z3_expr(self, m: Z3ExprMemo) -> z3.BitVecRef:
        return z3.Extract(self.upper, self.lower, self.bitvec.to_z3_bitvec(m))


class BitvecUnaryOp(Bitvec, ABC):
    bitvec: Bitvec

    def __init__(self, nid: int, sort: BitvecSort, bitvec: Bitvec, symbol: str = ""):
        super().__init__(nid, sort, symbol)
        self.bitvec = bitvec

    def to_new_z3_expr(self, m: Z3ExprMemo) -> Union[z3.BitVecRef, z3.BoolRef]:
        z3_bitvec_or_bool: Union[z3.BitVecRef, z3.BoolRef] = self.bitvec.to_z3_expr(m)
        if isinstance(z3_bitvec_or_bool, z3.BitVecRef):
            return self.z3_unary_bitvec_op(z3_bitvec_or_bool)
        return self.z3_unary_bool_op(z3_bitvec_or_bool)

    @abstractmethod
    def z3_unary_bitvec_op(
        self, z3_bitvec: z3.BitVecRef
    ) -> Union[z3.BitVecRef, z3.BoolRef]:
        raise NotImplementedError

    @abstractmethod
    def z3_unary_bool_op(self, z3_bool: z3.BoolRef) -> Union[z3.BitVecRef, z3.BoolRef]:
        raise NotImplementedError


class Not(BitvecUnaryOp):
    def z3_unary_bitvec_op(self, z3_bitvec: z3.BitVecRef) -> z3.BitVecRef:
        return ~z3_bitvec

    def z3_unary_bool_op(self, z3_bool: z3.BoolRef) -> z3.BoolRef:
        return z3.Not(z3_bool)


class Inc(BitvecUnaryOp):
    def z3_unary_bitvec_op(self, z3_bitvec: z3.BitVecRef) -> z3.BitVecRef:
        return z3_bitvec + 1

    def z3_unary_bool_op(self, z3_bool: z3.BoolRef) -> z3.BoolRef:
        return z3.Not(z3_bool)


class Dec(BitvecUnaryOp):
    def z3_unary_bitvec_op(self, z3_bitvec: z3.BitVecRef) -> z3.BitVecRef:
        return z3_bitvec - 1

    def z3_unary_bool_op(self, z3_bool: z3.BoolRef) -> z3.BoolRef:
        return z3.Not(z3_bool)


class Neg(BitvecUnaryOp):
    def z3_unary_bitvec_op(self, z3_bitvec: z3.BitVecRef) -> z3.BitVecRef:
        return -z3_bitvec

    def z3_unary_bool_op(self, z3_bool: z3.BoolRef) -> z3.BoolRef:
        return z3.Not(z3_bool)


class Redand(BitvecUnaryOp):
    def z3_unary_bitvec_op(self, z3_bitvec: z3.BitVecRef) -> z3.BitVecRef:
        return z3.BVRedAnd(z3_bitvec)

    def z3_unary_bool_op(self, z3_bool: z3.BoolRef) -> z3.BoolRef:
        return z3_bool


class Redor(BitvecUnaryOp):
    def z3_unary_bitvec_op(self, z3_bitvec: z3.BitVecRef) -> z3.BitVecRef:
        return z3.BVRedOr(z3_bitvec)

    def z3_unary_bool_op(self, z3_bool: z3.BoolRef) -> z3.BoolRef:
        return z3_bool


class Redxor(BitvecUnaryOp):
    def z3_unary_bitvec_op(self, z3_bitvec: z3.BitVecRef) -> z3.BitVecRef:
        z3_expr: z3.BitVecRef = z3.Extract(0, 0, z3_bitvec)
        for i in range(1, self.sort.width):
            z3_expr = z3_expr ^ z3.Extract(i, i, z3_bitvec)
        return z3_expr

    def z3_unary_bool_op(self, z3_bool: z3.BoolRef) -> z3.BoolRef:
        return z3_bool


class BitvecBinaryOp(Bitvec, ABC):
    bitvec1: Bitvec
    bitvec2: Bitvec

    def __init__(
        self,
        nid: int,
        sort: BitvecSort,
        bitvec1: Bitvec,
        bitvec2: Bitvec,
        symbol: str = "",
    ):
        super().__init__(nid, sort, symbol)
        self.bitvec1 = bitvec1
        self.bitvec2 = bitvec2

    def to_new_z3_expr(self, m: Z3ExprMemo) -> Union[z3.BitVecRef, z3.BoolRef]:
        z3_expr1: Union[z3.BitVecRef, z3.BoolRef] = self.bitvec1.to_z3_expr(m)
        z3_expr2: Union[z3.BitVecRef, z3.BoolRef] = self.bitvec2.to_z3_expr(m)
        if isinstance(z3_expr1, z3.BoolRef) or isinstance(z3_expr2, z3.BoolRef):
            return self.z3_bool_op(to_z3_bool(z3_expr1), to_z3_bool(z3_expr2))
        return self.z3_bitvec_op(z3_expr1, z3_expr2)

    @abstractmethod
    def z3_bitvec_op(
        self, z3_bitvec1: z3.BitVecRef, z3_bitvec2: z3.BitVecRef
    ) -> Union[z3.BitVecRef, z3.BoolRef]:
        raise NotImplementedError

    @abstractmethod
    def z3_bool_op(
        self, z3_bool1: z3.BoolRef, z3_bool2: z3.BoolRef
    ) -> Union[z3.BitVecRef, z3.BoolRef]:
        raise NotImplementedError


class Iff(BitvecBinaryOp):
    def z3_bitvec_op(
        self, z3_bitvec1: z3.BitVecRef, z3_bitvec2: z3.BitVecRef
    ) -> z3.BoolRef:
        return z3_bitvec1 == z3_bitvec2

    def z3_bool_op(self, z3_bool1: z3.BoolRef, z3_bool2: z3.BoolRef) -> z3.BoolRef:
        return z3_bool1 == z3_bool2


class Implies(BitvecBinaryOp):
    def z3_bitvec_op(
        self, z3_bitvec1: z3.BitVecRef, z3_bitvec2: z3.BitVecRef
    ) -> z3.BoolRef:
        return z3.Or(
            z3_bitvec1 == z3.BitVecVal(0, 1, z3_bitvec1.ctx),
            z3_bitvec2 == z3.BitVecVal(1, 1, z3_bitvec1.ctx),
            z3_bitvec1.ctx,
        )

    def z3_bool_op(self, z3_bool1: z3.BoolRef, z3_bool2: z3.BoolRef) -> z3.BoolRef:
        return z3.Implies(z3_bool1, z3_bool2)


class Equality(Bitvec, ABC):
    expr1: Expr
    expr2: Expr

    def __init__(
        self, nid: int, sort: BitvecSort, expr1: Expr, expr2, symbol: str = ""
    ):
        super().__init__(nid, sort, symbol)
        self.expr1 = expr1
        self.expr2 = expr2


class Eq(Equality):
    def to_new_z3_expr(self, m: Z3ExprMemo) -> z3.BoolRef:
        return to_z3_eq(self.expr1.to_z3_expr(m), self.expr2.to_z3_expr(m))


class Neq(Equality):
    def to_new_z3_expr(self, m: Z3ExprMemo) -> z3.BoolRef:
        z3_expr1: z3.ExprRef = self.expr1.to_z3_expr(m)
        z3_expr2: z3.ExprRef = self.expr2.to_z3_expr(m)
        if isinstance(z3_expr1, z3.BoolRef) or isinstance(z3_expr2, z3.BoolRef):
            return to_z3_bool(z3_expr1) != to_z3_bool(z3_expr2)
        return z3_expr1 != z3_expr2


class Sgt(BitvecBinaryOp):
    def z3_bitvec_op(
        self, z3_bitvec1: z3.BitVecRef, z3_bitvec2: z3.BitVecRef
    ) -> z3.BoolRef:
        return z3_bitvec1 > z3_bitvec2

    def z3_bool_op(self, z3_bool1: z3.BoolRef, z3_bool2: z3.BoolRef) -> z3.BoolRef:
        return z3.And(z3.Not(z3_bool1), z3_bool2, z3_bool1.ctx)


class Ugt(BitvecBinaryOp):
    def z3_bitvec_op(
        self, z3_bitvec1: z3.BitVecRef, z3_bitvec2: z3.BitVecRef
    ) -> z3.BoolRef:
        return z3.UGT(z3_bitvec1, z3_bitvec2)

    def z3_bool_op(self, z3_bool1: z3.BoolRef, z3_bool2: z3.BoolRef) -> z3.BoolRef:
        return z3.And(z3_bool1, z3.Not(z3_bool2), z3_bool1.ctx)


class Sgte(BitvecBinaryOp):
    def z3_bitvec_op(
        self, z3_bitvec1: z3.BitVecRef, z3_bitvec2: z3.BitVecRef
    ) -> z3.BoolRef:
        return z3_bitvec1 >= z3_bitvec2

    def z3_bool_op(self, z3_bool1: z3.BoolRef, z3_bool2: z3.BoolRef) -> z3.BoolRef:
        return z3.Implies(z3_bool1, z3_bool2)


class Ugte(BitvecBinaryOp):
    def z3_bitvec_op(
        self, z3_bitvec1: z3.BitVecRef, z3_bitvec2: z3.BitVecRef
    ) -> z3.BoolRef:
        return z3.UGE(z3_bitvec1, z3_bitvec2)

    def z3_bool_op(self, z3_bool1: z3.BoolRef, z3_bool2: z3.BoolRef) -> z3.BoolRef:
        return z3.Implies(z3_bool2, z3_bool1)


class Slt(BitvecBinaryOp):
    def z3_bitvec_op(
        self, z3_bitvec1: z3.BitVecRef, z3_bitvec2: z3.BitVecRef
    ) -> z3.BoolRef:
        return z3_bitvec1 < z3_bitvec2

    def z3_bool_op(self, z3_bool1: z3.BoolRef, z3_bool2: z3.BoolRef) -> z3.BoolRef:
        return z3.And(z3_bool1, z3.Not(z3_bool2), z3_bool1.ctx)


class Ult(BitvecBinaryOp):
    def z3_bitvec_op(
        self, z3_bitvec1: z3.BitVecRef, z3_bitvec2: z3.BitVecRef
    ) -> z3.BoolRef:
        return z3.ULT(z3_bitvec1, z3_bitvec2)

    def z3_bool_op(self, z3_bool1: z3.BoolRef, z3_bool2: z3.BoolRef) -> z3.BoolRef:
        return z3.And(z3.Not(z3_bool1), z3_bool2, z3_bool1.ctx)


class Slte(BitvecBinaryOp):
    def z3_bitvec_op(
        self, z3_bitvec1: z3.BitVecRef, z3_bitvec2: z3.BitVecRef
    ) -> z3.BoolRef:
        return z3_bitvec1 <= z3_bitvec2

    def z3_bool_op(self, z3_bool1: z3.BoolRef, z3_bool2: z3.BoolRef) -> z3.BoolRef:
        return z3.Implies(z3_bool2, z3_bool1)


class Ulte(BitvecBinaryOp):
    def z3_bitvec_op(
        self, z3_bitvec1: z3.BitVecRef, z3_bitvec2: z3.BitVecRef
    ) -> z3.BoolRef:
        return z3.ULE(z3_bitvec1, z3_bitvec2)

    def z3_bool_op(self, z3_bool1: z3.BoolRef, z3_bool2: z3.BoolRef) -> z3.BoolRef:
        return z3.Implies(z3_bool1, z3_bool2)


class And(BitvecBinaryOp):
    def z3_bitvec_op(
        self, z3_bitvec1: z3.BitVecRef, z3_bitvec2: z3.BitVecRef
    ) -> z3.BitVecRef:
        return z3_bitvec1 & z3_bitvec2

    def z3_bool_op(self, z3_bool1: z3.BoolRef, z3_bool2: z3.BoolRef) -> z3.BoolRef:
        return z3.And(z3_bool1, z3_bool2, z3_bool1.ctx)


class Nand(BitvecBinaryOp):
    def z3_bitvec_op(
        self, z3_bitvec1: z3.BitVecRef, z3_bitvec2: z3.BitVecRef
    ) -> z3.BitVecRef:
        return call_z3_mk_bv(z3.Z3_mk_bvnand, z3_bitvec1, z3_bitvec2)

    def z3_bool_op(self, z3_bool1: z3.BoolRef, z3_bool2: z3.BoolRef) -> z3.BoolRef:
        return z3.Not(z3.And(z3_bool1, z3_bool2, z3_bool1.ctx))


class Nor(BitvecBinaryOp):
    def z3_bitvec_op(
        self, z3_bitvec1: z3.BitVecRef, z3_bitvec2: z3.BitVecRef
    ) -> z3.BitVecRef:
        return call_z3_mk_bv(z3.Z3_mk_bvnor, z3_bitvec1, z3_bitvec2)

    def z3_bool_op(self, z3_bool1: z3.BoolRef, z3_bool2: z3.BoolRef) -> z3.BoolRef:
        return z3.Not(z3.Or(z3_bool1, z3_bool2, z3_bool1.ctx))


class Or(BitvecBinaryOp):
    def z3_bitvec_op(
        self, z3_bitvec1: z3.BitVecRef, z3_bitvec2: z3.BitVecRef
    ) -> z3.BitVecRef:
        return z3_bitvec1 | z3_bitvec2

    def z3_bool_op(self, z3_bool1: z3.BoolRef, z3_bool2: z3.BoolRef) -> z3.BoolRef:
        return z3.Or(z3_bool1, z3_bool2, z3_bool1.ctx)


class Xnor(BitvecBinaryOp):
    def z3_bitvec_op(
        self, z3_bitvec1: z3.BitVecRef, z3_bitvec2: z3.BitVecRef
    ) -> z3.BitVecRef:
        return call_z3_mk_bv(z3.Z3_mk_bvxnor, z3_bitvec1, z3_bitvec2)

    def z3_bool_op(self, z3_bool1: z3.BoolRef, z3_bool2: z3.BoolRef) -> z3.BoolRef:
        return z3.Not(z3.Xor(z3_bool1, z3_bool2))


class Xor(BitvecBinaryOp):
    def z3_bitvec_op(
        self, z3_bitvec1: z3.BitVecRef, z3_bitvec2: z3.BitVecRef
    ) -> z3.BitVecRef:
        return z3_bitvec1 ^ z3_bitvec2

    def z3_bool_op(self, z3_bool1: z3.BoolRef, z3_bool2: z3.BoolRef) -> z3.BoolRef:
        return z3.Xor(z3_bool1, z3_bool2)


class Sll(BitvecBinaryOp):
    def z3_bitvec_op(
        self, z3_bitvec1: z3.BitVecRef, z3_bitvec2: z3.BitVecRef
    ) -> z3.BitVecRef:
        return z3_bitvec1 << z3_bitvec2

    def z3_bool_op(self, z3_bool1: z3.BoolRef, z3_bool2: z3.BoolRef) -> z3.BoolRef:
        return z3.And(z3_bool1, z3.Not(z3_bool2), z3_bool1.ctx)


class Sra(BitvecBinaryOp):
    def z3_bitvec_op(
        self, z3_bitvec1: z3.BitVecRef, z3_bitvec2: z3.BitVecRef
    ) -> z3.BitVecRef:
        return z3_bitvec1 >> z3_bitvec2

    def z3_bool_op(self, z3_bool1: z3.BoolRef, z3_bool2: z3.BoolRef) -> z3.BoolRef:
        return z3_bool1


class Srl(BitvecBinaryOp):
    def z3_bitvec_op(
        self, z3_bitvec1: z3.BitVecRef, z3_bitvec2: z3.BitVecRef
    ) -> z3.BitVecRef:
        return z3.LShR(z3_bitvec1, z3_bitvec2)

    def z3_bool_op(self, z3_bool1: z3.BoolRef, z3_bool2: z3.BoolRef) -> z3.BoolRef:
        return z3.And(z3_bool1, z3.Not(z3_bool2), z3_bool1.ctx)


class Rol(BitvecBinaryOp):
    def z3_bitvec_op(
        self, z3_bitvec1: z3.BitVecRef, z3_bitvec2: z3.BitVecRef
    ) -> z3.BitVecRef:
        return z3.RotateLeft(z3_bitvec1, z3_bitvec2)

    def z3_bool_op(self, z3_bool1: z3.BoolRef, z3_bool2: z3.BoolRef) -> z3.BoolRef:
        return z3_bool1


class Ror(BitvecBinaryOp):
    def z3_bitvec_op(
        self, z3_bitvec1: z3.BitVecRef, z3_bitvec2: z3.BitVecRef
    ) -> z3.BitVecRef:
        return z3.RotateRight(z3_bitvec1, z3_bitvec2)

    def z3_bool_op(self, z3_bool1: z3.BoolRef, z3_bool2: z3.BoolRef) -> z3.BoolRef:
        return z3_bool1


class Add(BitvecBinaryOp):
    def z3_bitvec_op(
        self, z3_bitvec1: z3.BitVecRef, z3_bitvec2: z3.BitVecRef
    ) -> z3.BitVecRef:
        return z3_bitvec1 + z3_bitvec2

    def z3_bool_op(self, z3_bool1: z3.BoolRef, z3_bool2: z3.BoolRef) -> z3.BoolRef:
        return z3.Xor(z3_bool1, z3_bool2)


class Mul(BitvecBinaryOp):
    def z3_bitvec_op(
        self, z3_bitvec1: z3.BitVecRef, z3_bitvec2: z3.BitVecRef
    ) -> z3.BitVecRef:
        return z3_bitvec1 * z3_bitvec2

    def z3_bool_op(self, z3_bool1: z3.BoolRef, z3_bool2: z3.BoolRef) -> z3.BoolRef:
        return z3.And(z3_bool1, z3_bool2, z3_bool1.ctx)


class Sdiv(BitvecBinaryOp):
    def z3_bitvec_op(
        self, z3_bitvec1: z3.BitVecRef, z3_bitvec2: z3.BitVecRef
    ) -> z3.BitVecRef:
        return z3_bitvec1 / z3_bitvec2

    def z3_bool_op(self, z3_bool1: z3.BoolRef, z3_bool2: z3.BoolRef) -> z3.BoolRef:
        return z3.Implies(z3_bool2, z3_bool1)


class Udiv(BitvecBinaryOp):
    def z3_bitvec_op(
        self, z3_bitvec1: z3.BitVecRef, z3_bitvec2: z3.BitVecRef
    ) -> z3.BitVecRef:
        return z3.UDiv(z3_bitvec1, z3_bitvec2)

    def z3_bool_op(self, z3_bool1: z3.BoolRef, z3_bool2: z3.BoolRef) -> z3.BoolRef:
        return z3.Implies(z3_bool2, z3_bool1)


class Smod(BitvecBinaryOp):
    def z3_bitvec_op(
        self, z3_bitvec1: z3.BitVecRef, z3_bitvec2: z3.BitVecRef
    ) -> z3.BitVecRef:
        return z3_bitvec1 % z3_bitvec2

    def z3_bool_op(self, z3_bool1: z3.BoolRef, z3_bool2: z3.BoolRef) -> z3.BoolRef:
        return z3.And(z3_bool1, z3.Not(z3_bool2), z3_bool1.ctx)


class Srem(BitvecBinaryOp):
    def z3_bitvec_op(
        self, z3_bitvec1: z3.BitVecRef, z3_bitvec2: z3.BitVecRef
    ) -> z3.BitVecRef:
        return z3.SRem(z3_bitvec1, z3_bitvec2)

    def z3_bool_op(self, z3_bool1: z3.BoolRef, z3_bool2: z3.BoolRef) -> z3.BoolRef:
        return z3.And(z3_bool1, z3.Not(z3_bool2), z3_bool1.ctx)


class Urem(BitvecBinaryOp):
    def z3_bitvec_op(
        self, z3_bitvec1: z3.BitVecRef, z3_bitvec2: z3.BitVecRef
    ) -> z3.BitVecRef:
        return z3.URem(z3_bitvec1, z3_bitvec2)

    def z3_bool_op(self, z3_bool1: z3.BoolRef, z3_bool2: z3.BoolRef) -> z3.BoolRef:
        return z3.And(z3_bool1, z3.Not(z3_bool2), z3_bool1.ctx)


class Sub(BitvecBinaryOp):
    def z3_bitvec_op(
        self, z3_bitvec1: z3.BitVecRef, z3_bitvec2: z3.BitVecRef
    ) -> z3.BitVecRef:
        return z3_bitvec1 - z3_bitvec2

    def z3_bool_op(self, z3_bool1: z3.BoolRef, z3_bool2: z3.BoolRef) -> z3.BoolRef:
        return z3.Xor(z3_bool1, z3_bool2)


class Saddo(BitvecBinaryOp):
    def z3_bitvec_op(
        self, z3_bitvec1: z3.BitVecRef, z3_bitvec2: z3.BitVecRef
    ) -> z3.BoolRef:
        return z3.Not(
            z3.Or(
                z3.BVAddNoOverflow(z3_bitvec1, z3_bitvec2, True),
                z3.BVAddNoUnderflow(z3_bitvec1, z3_bitvec2),
                z3_bitvec1.ctx,
            )
        )

    def z3_bool_op(self, z3_bool1: z3.BoolRef, z3_bool2: z3.BoolRef) -> z3.BoolRef:
        return z3.BoolVal(False, z3_bool1.ctx)


class Uaddo(BitvecBinaryOp):
    def z3_bitvec_op(
        self, z3_bitvec1: z3.BitVecRef, z3_bitvec2: z3.BitVecRef
    ) -> z3.BoolRef:
        return z3.Not(z3.BVAddNoOverflow(z3_bitvec1, z3_bitvec2, False))

    def z3_bool_op(self, z3_bool1: z3.BoolRef, z3_bool2: z3.BoolRef) -> z3.BoolRef:
        return z3.And(z3_bool1, z3_bool2, z3_bool1.ctx)


class Sdivo(BitvecBinaryOp):
    def z3_bitvec_op(
        self, z3_bitvec1: z3.BitVecRef, z3_bitvec2: z3.BitVecRef
    ) -> z3.BoolRef:
        return z3.Not(z3.BVSDivNoOverflow(z3_bitvec1, z3_bitvec2))

    def z3_bool_op(self, z3_bool1: z3.BoolRef, z3_bool2: z3.BoolRef) -> z3.BoolRef:
        return z3.And(z3_bool1, z3_bool2, z3_bool1.ctx)


class Udivo(BitvecBinaryOp):
    def z3_bitvec_op(
        self, z3_bitvec1: z3.BitVecRef, z3_bitvec2: z3.BitVecRef
    ) -> z3.BoolRef:
        return z3.BoolVal(False, z3_bitvec1.ctx)

    def z3_bool_op(self, z3_bool1: z3.BoolRef, z3_bool2: z3.BoolRef) -> z3.BoolRef:
        return z3.BoolVal(False, z3_bool1.ctx)


class Smulo(BitvecBinaryOp):
    def z3_bitvec_op(
        self, z3_bitvec1: z3.BitVecRef, z3_bitvec2: z3.BitVecRef
    ) -> z3.BoolRef:
        return z3.Not(
            z3.Or(
                z3.BVMulNoOverflow(z3_bitvec1, z3_bitvec2, True),
                z3.BVMulNoUnderflow(z3_bitvec1, z3_bitvec2),
                z3_bitvec1.ctx,
            )
        )

    def z3_bool_op(self, z3_bool1: z3.BoolRef, z3_bool2: z3.BoolRef) -> z3.BoolRef:
        return z3.BoolVal(False, z3_bool1.ctx)


class Umulo(BitvecBinaryOp):
    def z3_bitvec_op(
        self, z3_bitvec1: z3.BitVecRef, z3_bitvec2: z3.BitVecRef
    ) -> z3.BoolRef:
        return z3.Not(z3.BVMulNoOverflow(z3_bitvec1, z3_bitvec2, False))

    def z3_bool_op(self, z3_bool1: z3.BoolRef, z3_bool2: z3.BoolRef) -> z3.BoolRef:
        return z3.BoolVal(False, z3_bool1.ctx)


class Ssubo(BitvecBinaryOp):
    def z3_bitvec_op(
        self, z3_bitvec1: z3.BitVecRef, z3_bitvec2: z3.BitVecRef
    ) -> z3.BoolRef:
        return z3.Not(
            z3.Or(
                z3.BVSubNoUnderflow(z3_bitvec1, z3_bitvec2, True),
                z3.BVSubNoOverflow(z3_bitvec1, z3_bitvec2),
                z3_bitvec1.ctx,
            )
        )

    def z3_bool_op(self, z3_bool1: z3.BoolRef, z3_bool2: z3.BoolRef) -> z3.BoolRef:
        return z3.BoolVal(False, z3_bool1.ctx)


class Usubo(BitvecBinaryOp):
    def z3_bitvec_op(
        self, z3_bitvec1: z3.BitVecRef, z3_bitvec2: z3.BitVecRef
    ) -> z3.BoolRef:
        return z3.Not(z3.BVSubNoUnderflow(z3_bitvec1, z3_bitvec2, False))

    def z3_bool_op(self, z3_bool1: z3.BoolRef, z3_bool2: z3.BoolRef) -> z3.BoolRef:
        return z3.And(z3_bool1, z3.Not(z3_bool2), z3_bool1.ctx)


class Concat(BitvecBinaryOp):
    def to_new_z3_expr(self, m: Z3ExprMemo) -> Union[z3.BitVecRef, z3.BoolRef]:
        return z3.Concat(self.bitvec1.to_z3_bitvec(m), self.bitvec2.to_z3_bitvec(m))

    def z3_bitvec_op(
        self, z3_bitvec1: z3.BitVecRef, z3_bitvec2: z3.BitVecRef
    ) -> z3.BitVecRef:
        raise ValueError

    def z3_bool_op(self, z3_bool1: z3.BoolRef, z3_bool2: z3.BoolRef) -> z3.BitVecRef:
        raise ValueError


class Read:
    array: Array
    index_expr: Expr

    def __init__(self, array: Array, index_expr: Expr):
        self.array = array
        self.index_expr = index_expr


class BitvecRead(Bitvec, Read):
    def __init__(
        self,
        nid: int,
        sort: BitvecSort,
        array: Array,
        index_expr: Expr,
        symbol: str = "",
    ):
        Bitvec.__init__(self, nid, sort, symbol)
        Read.__init__(self, array, index_expr)

    def to_new_z3_expr(self, m: Z3ExprMemo) -> z3.BitVecRef:
        return self.array.to_z3_expr(m)[
            try_bitvec_to_bool(self.index_expr.to_z3_expr(m))
        ]


class ArrayRead(Array, Read):
    def __init__(
        self,
        nid: int,
        sort: ArraySort,
        array: Array,
        index_expr: Expr,
        symbol: str = "",
    ):
        Array.__init__(self, nid, sort, symbol)
        Read.__init__(self, array, index_expr)

    def to_new_z3_expr(self, m: Z3ExprMemo) -> z3.ArrayRef:
        return self.array.to_z3_expr(m)[
            try_bitvec_to_bool(self.index_expr.to_z3_expr(m))
        ]


class Ite:
    cond_bitvec: Bitvec
    then_expr: Expr
    else_expr: Expr

    def __init__(self, cond_bitvec: Bitvec, then_expr: Expr, else_expr: Expr):
        self.cond_bitvec = cond_bitvec
        self.then_expr = then_expr
        self.else_expr = else_expr


class ArrayIte(Array, Ite):
    then_expr: Array
    else_expr: Array

    def __init__(
        self,
        nid: int,
        sort: ArraySort,
        cond_bitvec: Bitvec,
        then_expr: Array,
        else_expr: Array,
        symbol: str = "",
    ):
        Array.__init__(self, nid, sort, symbol)
        Ite.__init__(self, cond_bitvec, then_expr, else_expr)

    def to_new_z3_expr(self, m: Z3ExprMemo) -> z3.ArrayRef:
        cond_z3_bitvec: z3.BoolRef = self.cond_bitvec.to_z3_bool(m)
        if z3.is_true(cond_z3_bitvec):
            return self.then_expr.to_z3_expr(m)
        if z3.is_false(cond_z3_bitvec):
            return self.else_expr.to_z3_expr(m)
        return z3.If(
            self.cond_bitvec.to_z3_bool(m),
            self.then_expr.to_z3_expr(m),
            self.else_expr.to_z3_expr(m),
        )


class BitvecIte(Bitvec, Ite):
    then_expr: Bitvec
    else_expr: Bitvec

    def __init__(
        self,
        nid: int,
        sort: BitvecSort,
        cond_bitvec: Bitvec,
        then_expr: Bitvec,
        else_expr: Bitvec,
        symbol: str = "",
    ):
        Bitvec.__init__(self, nid, sort, symbol)
        Ite.__init__(self, cond_bitvec, then_expr, else_expr)

    def to_new_z3_expr(self, m: Z3ExprMemo) -> Union[z3.BitVecRef, z3.BoolRef]:
        cond_z3_bitvec: z3.BoolRef = self.cond_bitvec.to_z3_bool(m)
        then_z3_expr: Union[z3.BitVecRef, z3.BoolRef] = self.then_expr.to_z3_expr(m)
        else_z3_expr: Union[z3.BitVecRef, z3.BoolRef] = self.else_expr.to_z3_expr(m)

        if z3.is_true(cond_z3_bitvec):
            return then_z3_expr
        if z3.is_false(cond_z3_bitvec):
            return else_z3_expr

        if isinstance(then_z3_expr, z3.BoolRef) or isinstance(else_z3_expr, z3.BoolRef):
            then_z3_expr = to_z3_bool(then_z3_expr)
            else_z3_expr = to_z3_bool(else_z3_expr)
        return z3.If(self.cond_bitvec.to_z3_bool(m), then_z3_expr, else_z3_expr)


class Write(Array):
    array: Array
    index_expr: Expr
    element_expr: Expr

    def __init__(
        self,
        nid: int,
        sort: ArraySort,
        array: Array,
        index_expr: Expr,
        element_expr: Expr,
        symbol: str = "",
    ):
        super().__init__(nid, sort, symbol)
        self.array = array
        self.index_expr = index_expr
        self.element_expr = element_expr

    def to_new_z3_expr(self, m: Z3ExprMemo) -> z3.ArrayRef:
        return z3.Update(
            self.array.to_z3_expr(m),
            try_bitvec_to_bool(self.index_expr.to_z3_expr(m)),
            try_bitvec_to_bool(self.element_expr.to_z3_expr(m)),
        )


class BitvecInit(Node):
    nid: int
    sort: BitvecSort
    state: BitvecState
    bitvec: Bitvec

    def __init__(
        self,
        nid: int,
        sort: BitvecSort,
        state: BitvecState,
        bitvec: Bitvec,
        symbol: str = "",
    ):
        super().__init__(symbol)
        self.nid = nid
        self.sort = sort
        self.state = state
        self.bitvec = bitvec


class ArrayInit(Node):
    nid: int
    sort: ArraySort
    state: ArrayState
    array: Array

    def __init__(
        self,
        nid: int,
        sort: ArraySort,
        state: ArrayState,
        bitvec: Bitvec,
        symbol: str = "",
    ):
        super().__init__(symbol)
        self.nid = nid
        self.sort = sort
        self.state = state
        self.bitvec = bitvec


class BitvecNext(Node):
    nid: int
    sort: BitvecSort
    state: BitvecState
    bitvec: Bitvec

    def __init__(
        self,
        nid: int,
        sort: BitvecSort,
        state: BitvecState,
        bitvec: Bitvec,
        symbol: str = "",
    ):
        super().__init__(symbol)
        self.nid = nid
        self.sort = sort
        self.state = state
        self.bitvec = bitvec


class ArrayNext(Node):
    nid: int
    sort: ArraySort
    state: ArrayState
    array: Array

    def __init__(
        self,
        nid: int,
        sort: ArraySort,
        state: ArrayState,
        bitvec: Bitvec,
        symbol: str = "",
    ):
        super().__init__(symbol)
        self.nid = nid
        self.sort = sort
        self.state = state
        self.bitvec = bitvec


class Bad(Node):
    nid: int
    bitvec: Bitvec

    def __init__(self, nid: int, bitvec: Bitvec, symbol: str = ""):
        super().__init__(symbol)
        self.nid = nid
        self.bitvec = bitvec


class Constraint(Node):
    nid: int
    bitvec: Bitvec

    def __init__(self, nid: int, bitvec: Bitvec, symbol: str = ""):
        super().__init__(symbol)
        self.nid = nid
        self.bitvec = bitvec


class Fair(Node):
    nid: int
    expr: Expr

    def __init__(self, nid: int, expr: Expr, symbol: str = ""):
        super().__init__(symbol)
        self.nid = nid
        self.expr = expr


class Output(Node):
    nid: int
    expr: Expr

    def __init__(self, nid: int, expr: Expr, symbol: str = ""):
        super().__init__(symbol)
        self.nid = nid
        self.expr = expr


class Justice(Node):
    nid: int
    n: int
    expr_list: List[Expr]

    def __init__(self, nid: int, n: int, expr_list: List[Expr], symbol: str = ""):
        super().__init__(symbol)
        self.nid = nid
        self.n = n
        self.expr_list = expr_list


class Ts:
    ctx: z3.Context
    pre_vars: List[z3.ExprRef]
    post_vars: List[z3.ExprRef]
    inputs: List[z3.ExprRef]
    init: z3.BoolRef
    tr: z3.BoolRef
    bads: List[z3.BoolRef]
    constraints: List[z3.ExprRef]

    def __init__(self, ctx: z3.Context = z3.main_ctx()):
        self.ctx = ctx
        self.pre_vars = []
        self.post_vars = []
        self.inputs = []
        self.init = z3.BoolVal(True, ctx)
        self.tr = z3.BoolVal(True, ctx)
        self.bads = []
        self.constraints = []

    def put_var(
        self, sort: z3.SortRef, prefix: str = ""
    ) -> Tuple[z3.ExprRef, z3.ExprRef]:
        pre_z3_expr: z3.ExprRef = z3.FreshConst(sort, "vi" + prefix)
        post_z3_expr: z3.ExprRef = z3.FreshConst(sort, "vo" + prefix)
        self.pre_vars.append(pre_z3_expr)
        self.post_vars.append(post_z3_expr)
        return pre_z3_expr, post_z3_expr

    def put_input(self, sort: z3.SortRef, prefix: str = "") -> z3.ExprRef:
        z3_expr: z3.ExprRef = z3.FreshConst(sort, "i" + prefix)
        self.inputs.append(z3_expr)
        return z3_expr

    def sig(self) -> List[z3.SortRef]:
        return [v.sort() for v in self.post_vars]

    def inv(self) -> z3.FuncDeclRef:
        return z3.Function("Inv", *self.sig(), z3.BoolSort(self.ctx))

    def vars(self) -> List[z3.ExprRef]:
        return self.pre_vars + self.post_vars

    def vars_and_inputs(self) -> List[z3.ExprRef]:
        return self.vars() + self.inputs

    def bad(self) -> z3.ExprRef:
        self.reduce_bads()
        if self.bads:
            return self.bads[0]
        return z3.Bool(False, self.ctx)

    def reduce_bads(self) -> None:
        if len(self.bads) > 1:
            self.bads = [z3.Or(self.bads, self.ctx)]

    def reduce_constraints(self, method: int = 0) -> None:
        if self.constraints:
            # (init(v), tr(v, in, v'), bad(v)) with constraint c(v) is equiv to
            # init(v) and c(v), tr(v, in, v') and c(v), bad(v) and c(v)
            # (init(v), tr(v, in, v'), bad(v)) with constraint c(v, in) is equiv to
            # init(v), tr(v, in, v') and c(v, in), bad(v)
            if method == 0:
                self.init = z3.And(self.init, *self.constraints, self.ctx)
                self.tr = z3.And(self.tr, *self.constraints, self.ctx)
                self.bads = [z3.And(self.bad(), *self.constraints, self.ctx)]
            elif method == 1:
                self.tr = z3.And(self.tr, *self.constraints, self.ctx)
            else:
                raise NotImplementedError
            self.constraints = []

    def simplify(self) -> None:
        self.init = z3.simplify(self.init)
        self.tr = z3.simplify(self.tr)
        self.bads = [z3.simplify(bad) for bad in self.bads]


def generate_vc(ts: Ts) -> Tuple[z3.ExprRef, z3.ExprRef, z3.ExprRef, z3.FuncDeclRef]:
    if len(ts.bads) > 1 or ts.constraints:
        raise NotImplementedError
    vars_and_inputs: List[z3.ExprRef] = ts.vars_and_inputs()
    inv: z3.FuncDeclRef = ts.inv()
    inv_pre: z3.ExprRef = inv(*ts.pre_vars)
    inv_post: z3.ExprRef = inv(*ts.post_vars)

    init: z3.ExprRef = z3.Implies(ts.init, inv_pre)
    tr: z3.ExprRef = z3.Implies(z3.And(inv_pre, ts.tr, ts.ctx), inv_post)
    bad: z3.ExprRef = z3.Implies(z3.And(inv_pre, ts.bad(), ts.ctx), False)

    if vars_and_inputs:
        init = z3.ForAll(vars_and_inputs, init)
        tr = z3.ForAll(vars_and_inputs, tr)
        bad = z3.ForAll(vars_and_inputs, bad)

    return init, tr, bad, inv


def generate_chc_str(
    init: z3.ExprRef, tr: z3.ExprRef, bad: z3.ExprRef, inv: z3.FuncDeclRef
) -> List[str]:
    return [
        "(set-logic HORN)\n",
        "(set-option :fp.engine spacer)\n",
        "{:s}\n".format(inv.sexpr()),
        "(assert {:s})\n".format(init.sexpr()),
        "(assert {:s})\n".format(tr.sexpr()),
        "(assert {:s})\n".format(bad.sexpr()),
        "(check-sat)\n",
        "(exit)\n",
    ]


def bmc(
    init: z3.ExprRef,
    tr: z3.ExprRef,
    bad: z3.ExprRef,
    n: int,
    pre_vars: List[z3.ExprRef],
    post_vars: List[z3.ExprRef],
    inputs: List[z3.ExprRef],
) -> z3.ExprRef:
    z3_exprs: List[z3.ExprRef] = [init, tr]

    vars1: List[z3.ExprRef] = pre_vars.copy()
    vars2: List[z3.ExprRef] = post_vars.copy()
    ips: List[z3.ExprRef] = inputs.copy()
    all_vars: List[z3.ExprRef] = pre_vars + post_vars + inputs

    _: int
    for _ in range(n - 1):
        new_vars: List[z3.ExprRef] = []
        new_ips: List[z3.ExprRef] = []
        var: z3.ExprRef

        for var in vars2:
            new_vars.append(z3.FreshConst(var.sort(), var.sexpr().split("!")[0]))
        for var in inputs:
            new_ips.append(z3.FreshConst(var.sort(), var.sexpr().split("!")[0]))

        tr = z3.substitute(
            tr, *zip(vars2, new_vars), *zip(vars1, vars2), *zip(ips, new_ips)
        )
        vars1, vars2, ips = vars2, new_vars, new_ips
        z3_exprs.append(tr)
        all_vars += new_vars + new_ips

    z3_expr = z3.And(z3_exprs + [z3.substitute(bad, *zip(pre_vars, vars2))], init.ctx)

    if all_vars:
        return z3.ForAll(all_vars, z3_expr)
    return z3_expr


class Btor2Parser:
    ctx: z3.Context
    bitvec_sort_table: Dict[int, BitvecSort]
    array_sort_table: Dict[int, ArraySort]
    bitvec_state_table: Dict[int, BitvecState]
    array_state_table: Dict[int, ArrayState]
    bitvec_input_table: Dict[int, BitvecInput]
    array_input_table: Dict[int, ArrayInput]
    bitvec_table: Dict[int, Bitvec]
    array_table: Dict[int, Array]
    bad_list: List[Bad]
    constraint_list: List[Constraint]
    fair_list: List[Fair]
    output_list: List[Output]
    justice_list: List[Justice]

    def __init__(self, ctx: z3.Context = z3.main_ctx()):
        self.ctx = ctx
        self.bitvec_sort_table = {}
        self.bitvec_state_table = {}
        self.bitvec_input_table = {}
        self.bitvec_table = {}
        self.array_sort_table = {}
        self.array_input_table = {}
        self.array_state_table = {}
        self.array_table = {}
        self.bad_list = []
        self.constraint_list = []
        self.fair_list = []
        self.output_list = []
        self.justice_list = []

    def get_sort(self, s: Union[int, str]) -> Sort:
        sid: int = int(s)
        return (
            self.bitvec_sort_table[sid]
            if sid in self.bitvec_sort_table
            else self.array_sort_table[sid]
        )

    def get_bitvec_sort(self, s: Union[int, str]) -> BitvecSort:
        return self.bitvec_sort_table[int(s)]

    def get_array_sort(self, s: Union[int, str]) -> ArraySort:
        return self.array_sort_table[int(s)]

    def get_expr(self, n: Union[int, str]) -> Expr:
        nid: int = int(n)
        if nid < 0:
            return Minus(self.bitvec_table[-nid])
        return (
            self.bitvec_table[nid]
            if nid in self.bitvec_table
            else self.array_table[nid]
        )

    def get_bitvec_state(self, n: Union[int, str]) -> BitvecState:
        return self.bitvec_state_table[int(n)]

    def get_array_state(self, n: Union[int, str]) -> ArrayState:
        return self.array_state_table[int(n)]

    def get_bitvec(self, n: Union[int, str]) -> Bitvec:
        nid: int = int(n)
        if nid < 0:
            return Minus(self.bitvec_table[-nid])
        return self.bitvec_table[nid]

    def get_array(self, n: Union[int, str]) -> Array:
        return self.array_table[int(n)]

    def parse(self, source: TextIO) -> None:
        for line in source:
            line_left: str
            _: str
            line_left, _, _ = line.partition(";")
            tokens: List[str] = line_left.split()

            if len(tokens) == 0:
                continue

            name: str = tokens[1]
            if name == "sort":
                sid: int = int(tokens[0])
                if tokens[2] == "array":
                    self.array_sort_table[sid] = ArraySort(
                        sid, self.get_sort(tokens[3]), self.get_sort(tokens[4])
                    )
                elif tokens[2] == "bitvec":
                    self.bitvec_sort_table[sid] = BitvecSort(sid, int(tokens[3]))
                continue

            nid: int = int(tokens[0])

            if name == "bad":
                self.bad_list.append(Bad(nid, self.get_bitvec(tokens[2])))
                continue
            if name == "constraint":
                self.constraint_list.append(Constraint(nid, self.get_bitvec(tokens[2])))
                continue
            if name == "fair":
                self.fair_list.append(Fair(nid, self.get_expr(tokens[2])))
                continue
            if name == "output":
                self.output_list.append(Output(nid, self.get_expr(tokens[2])))
                continue
            if name == "justice":
                n: int = int(tokens[2])
                self.justice_list.append(
                    Justice(nid, n, [self.get_expr(x) for x in tokens[3 : 3 + n]])
                )
                continue

            # noinspection DuplicatedCode
            if name == "read":
                read_sid: int = int(tokens[2])
                if read_sid in self.bitvec_sort_table:
                    self.bitvec_table[nid] = BitvecRead(
                        nid,
                        self.get_bitvec_sort(read_sid),
                        self.get_array(tokens[3]),
                        self.get_expr(tokens[4]),
                    )
                elif read_sid in self.array_sort_table:
                    self.array_table[nid] = ArrayRead(
                        nid,
                        self.get_array_sort(read_sid),
                        self.get_array(tokens[3]),
                        self.get_expr(tokens[4]),
                    )
                continue
            if name == "state":
                state_sid: int = int(tokens[2])
                if state_sid in self.bitvec_sort_table:
                    bitvec_state: BitvecState = BitvecState(
                        nid, self.get_bitvec_sort(state_sid)
                    )
                    self.bitvec_state_table[nid] = self.bitvec_table[nid] = bitvec_state
                elif state_sid in self.array_sort_table:
                    array_state: ArrayState = ArrayState(
                        nid, self.get_array_sort(state_sid)
                    )
                    self.array_state_table[nid] = self.array_table[nid] = array_state
                continue
            if name == "input":
                input_sid: int = int(tokens[2])
                if input_sid in self.bitvec_sort_table:
                    bitvec_input: BitvecInput = BitvecInput(
                        nid, self.get_bitvec_sort(input_sid)
                    )
                    self.bitvec_input_table[nid] = self.bitvec_table[nid] = bitvec_input
                elif input_sid in self.array_sort_table:
                    array_input: ArrayInput = ArrayInput(
                        nid, self.get_array_sort(input_sid)
                    )
                    self.array_input_table[nid] = self.array_table[nid] = array_input
                continue
            if name == "init":
                init_sid: int = int(tokens[2])
                if init_sid in self.bitvec_sort_table:
                    self.get_bitvec_state(tokens[3]).init = self.get_bitvec(tokens[4])
                elif init_sid in self.array_sort_table:
                    self.get_array_state(tokens[3]).init = self.get_expr(tokens[4])
                continue
            if name == "next":
                next_sid: int = int(tokens[2])
                if next_sid in self.bitvec_sort_table:
                    self.get_bitvec_state(tokens[3]).next = self.get_bitvec(tokens[4])
                elif next_sid in self.array_sort_table:
                    self.get_array_state(tokens[3]).next = self.get_array(tokens[4])
                continue
            if name == "write":
                self.array_table[nid] = Write(
                    nid,
                    self.get_array_sort(int(tokens[2])),
                    self.get_array(tokens[3]),
                    self.get_expr(tokens[4]),
                    self.get_expr(tokens[5]),
                )
                continue
            if name == "ite":
                ite_sid: int = int(tokens[2])
                if ite_sid in self.bitvec_sort_table:
                    self.bitvec_table[nid] = BitvecIte(
                        nid,
                        self.bitvec_sort_table[ite_sid],
                        self.get_bitvec(tokens[3]),
                        self.get_bitvec(tokens[4]),
                        self.get_bitvec(tokens[5]),
                    )
                elif ite_sid in self.array_sort_table:
                    self.array_table[nid] = ArrayIte(
                        nid,
                        self.array_sort_table[ite_sid],
                        self.get_bitvec(tokens[3]),
                        self.get_array(tokens[4]),
                        self.get_array(tokens[5]),
                    )
                continue

            sort: BitvecSort = self.get_bitvec_sort(tokens[2])
            if name == "one":
                self.bitvec_table[nid] = One(nid, sort)
            elif name == "ones":
                self.bitvec_table[nid] = Ones(nid, sort)
            elif name == "zero":
                self.bitvec_table[nid] = Zero(nid, sort)
            elif name == "const":
                self.bitvec_table[nid] = Const(nid, sort, tokens[3])
            elif name == "constd":
                self.bitvec_table[nid] = Constd(nid, sort, tokens[3])
            elif name == "consth":
                self.bitvec_table[nid] = Consth(nid, sort, tokens[3])
            elif name == "sext":
                self.bitvec_table[nid] = Sext(
                    nid, sort, self.get_bitvec(tokens[3]), int(tokens[4])
                )
            elif name == "uext":
                self.bitvec_table[nid] = Uext(
                    nid, sort, self.get_bitvec(tokens[3]), int(tokens[4])
                )
            elif name == "slice":
                self.bitvec_table[nid] = Slice(
                    nid,
                    sort,
                    self.get_bitvec(tokens[3]),
                    int(tokens[4]),
                    int(tokens[5]),
                )
            elif name == "not":
                self.bitvec_table[nid] = Not(nid, sort, self.get_bitvec(tokens[3]))
            elif name == "inc":
                self.bitvec_table[nid] = Inc(nid, sort, self.get_bitvec(tokens[3]))
            elif name == "dec":
                self.bitvec_table[nid] = Dec(nid, sort, self.get_bitvec(tokens[3]))
            elif name == "neg":
                self.bitvec_table[nid] = Neg(nid, sort, self.get_bitvec(tokens[3]))
            elif name == "redand":
                self.bitvec_table[nid] = Redand(nid, sort, self.get_bitvec(tokens[3]))
            elif name == "redor":
                self.bitvec_table[nid] = Redor(nid, sort, self.get_bitvec(tokens[3]))
            elif name == "redxor":
                self.bitvec_table[nid] = Redxor(nid, sort, self.get_bitvec(tokens[3]))
            elif name == "iff":
                self.bitvec_table[nid] = Iff(
                    nid, sort, self.get_bitvec(tokens[3]), self.get_bitvec(tokens[4])
                )
            elif name == "implies":
                self.bitvec_table[nid] = Implies(
                    nid, sort, self.get_bitvec(tokens[3]), self.get_bitvec(tokens[4])
                )
            elif name == "eq":
                self.bitvec_table[nid] = Eq(
                    nid, sort, self.get_expr(tokens[3]), self.get_expr(tokens[4])
                )
            elif name == "neq":
                self.bitvec_table[nid] = Neq(
                    nid, sort, self.get_expr(tokens[3]), self.get_expr(tokens[4])
                )
            elif name == "sgt":
                self.bitvec_table[nid] = Sgt(
                    nid, sort, self.get_bitvec(tokens[3]), self.get_bitvec(tokens[4])
                )
            elif name == "ugt":
                self.bitvec_table[nid] = Ugt(
                    nid, sort, self.get_bitvec(tokens[3]), self.get_bitvec(tokens[4])
                )
            elif name == "sgte":
                self.bitvec_table[nid] = Sgte(
                    nid, sort, self.get_bitvec(tokens[3]), self.get_bitvec(tokens[4])
                )
            elif name == "ugte":
                self.bitvec_table[nid] = Ugte(
                    nid, sort, self.get_bitvec(tokens[3]), self.get_bitvec(tokens[4])
                )
            elif name == "slt":
                self.bitvec_table[nid] = Slt(
                    nid, sort, self.get_bitvec(tokens[3]), self.get_bitvec(tokens[4])
                )
            elif name == "ult":
                self.bitvec_table[nid] = Ult(
                    nid, sort, self.get_bitvec(tokens[3]), self.get_bitvec(tokens[4])
                )
            elif name == "slte":
                self.bitvec_table[nid] = Slte(
                    nid, sort, self.get_bitvec(tokens[3]), self.get_bitvec(tokens[4])
                )
            elif name == "ulte":
                self.bitvec_table[nid] = Ulte(
                    nid, sort, self.get_bitvec(tokens[3]), self.get_bitvec(tokens[4])
                )
            elif name == "and":
                self.bitvec_table[nid] = And(
                    nid, sort, self.get_bitvec(tokens[3]), self.get_bitvec(tokens[4])
                )
            elif name == "nand":
                self.bitvec_table[nid] = Nand(
                    nid, sort, self.get_bitvec(tokens[3]), self.get_bitvec(tokens[4])
                )
            elif name == "nor":
                self.bitvec_table[nid] = Nor(
                    nid, sort, self.get_bitvec(tokens[3]), self.get_bitvec(tokens[4])
                )
            elif name == "or":
                self.bitvec_table[nid] = Or(
                    nid, sort, self.get_bitvec(tokens[3]), self.get_bitvec(tokens[4])
                )
            elif name == "xnor":
                self.bitvec_table[nid] = Xnor(
                    nid, sort, self.get_bitvec(tokens[3]), self.get_bitvec(tokens[4])
                )
            elif name == "xor":
                self.bitvec_table[nid] = Xor(
                    nid, sort, self.get_bitvec(tokens[3]), self.get_bitvec(tokens[4])
                )
            elif name == "rol":
                self.bitvec_table[nid] = Rol(
                    nid, sort, self.get_bitvec(tokens[3]), self.get_bitvec(tokens[4])
                )
            elif name == "ror":
                self.bitvec_table[nid] = Ror(
                    nid, sort, self.get_bitvec(tokens[3]), self.get_bitvec(tokens[4])
                )
            elif name == "sll":
                self.bitvec_table[nid] = Sll(
                    nid, sort, self.get_bitvec(tokens[3]), self.get_bitvec(tokens[4])
                )
            elif name == "sra":
                self.bitvec_table[nid] = Sra(
                    nid, sort, self.get_bitvec(tokens[3]), self.get_bitvec(tokens[4])
                )
            elif name == "srl":
                self.bitvec_table[nid] = Srl(
                    nid, sort, self.get_bitvec(tokens[3]), self.get_bitvec(tokens[4])
                )
            elif name == "add":
                self.bitvec_table[nid] = Add(
                    nid, sort, self.get_bitvec(tokens[3]), self.get_bitvec(tokens[4])
                )
            elif name == "mul":
                self.bitvec_table[nid] = Mul(
                    nid, sort, self.get_bitvec(tokens[3]), self.get_bitvec(tokens[4])
                )
            elif name == "sdiv":
                self.bitvec_table[nid] = Sdiv(
                    nid, sort, self.get_bitvec(tokens[3]), self.get_bitvec(tokens[4])
                )
            elif name == "udiv":
                self.bitvec_table[nid] = Udiv(
                    nid, sort, self.get_bitvec(tokens[3]), self.get_bitvec(tokens[4])
                )
            elif name == "smod":
                self.bitvec_table[nid] = Smod(
                    nid, sort, self.get_bitvec(tokens[3]), self.get_bitvec(tokens[4])
                )
            elif name == "srem":
                self.bitvec_table[nid] = Srem(
                    nid, sort, self.get_bitvec(tokens[3]), self.get_bitvec(tokens[4])
                )
            elif name == "urem":
                self.bitvec_table[nid] = Urem(
                    nid, sort, self.get_bitvec(tokens[3]), self.get_bitvec(tokens[4])
                )
            elif name == "sub":
                self.bitvec_table[nid] = Sub(
                    nid, sort, self.get_bitvec(tokens[3]), self.get_bitvec(tokens[4])
                )
            elif name == "saddo":
                self.bitvec_table[nid] = Saddo(
                    nid, sort, self.get_bitvec(tokens[3]), self.get_bitvec(tokens[4])
                )
            elif name == "uaddo":
                self.bitvec_table[nid] = Uaddo(
                    nid, sort, self.get_bitvec(tokens[3]), self.get_bitvec(tokens[4])
                )
            elif name == "sdivo":
                self.bitvec_table[nid] = Sdivo(
                    nid, sort, self.get_bitvec(tokens[3]), self.get_bitvec(tokens[4])
                )
            elif name == "udivo":
                self.bitvec_table[nid] = Udivo(
                    nid, sort, self.get_bitvec(tokens[3]), self.get_bitvec(tokens[4])
                )
            elif name == "smulo":
                self.bitvec_table[nid] = Smulo(
                    nid, sort, self.get_bitvec(tokens[3]), self.get_bitvec(tokens[4])
                )
            elif name == "umulo":
                self.bitvec_table[nid] = Umulo(
                    nid, sort, self.get_bitvec(tokens[3]), self.get_bitvec(tokens[4])
                )
            elif name == "ssubo":
                self.bitvec_table[nid] = Ssubo(
                    nid, sort, self.get_bitvec(tokens[3]), self.get_bitvec(tokens[4])
                )
            elif name == "usubo":
                self.bitvec_table[nid] = Usubo(
                    nid, sort, self.get_bitvec(tokens[3]), self.get_bitvec(tokens[4])
                )
            elif name == "concat":
                self.bitvec_table[nid] = Concat(
                    nid, sort, self.get_bitvec(tokens[3]), self.get_bitvec(tokens[4])
                )

    def to_ts(self) -> Ts:
        ts: Ts = Ts(self.ctx)

        nid: int
        bitvec_input: BitvecInput
        bitvec_state: BitvecState
        array_input: ArrayInput
        array_state: ArrayState
        z3_expr: z3.ExprRef

        m: Z3ExprMemo = Z3ExprMemo(self.ctx)
        m_post: Z3ExprMemo = Z3ExprMemo(self.ctx)
        z3_inits: List[z3.BoolRef] = []
        z3_nexts: List[z3.BoolRef] = []

        for nid, bitvec_input in self.bitvec_input_table.items():
            z3_expr = ts.put_input(
                bitvec_input.sort.to_z3_sort(self.ctx), bitvec_input.symbol
            )
            if isinstance(z3_expr, (z3.BitVecRef, z3.BoolRef)):
                m.z3_bitvec_or_bool_m[nid] = z3_expr
            else:
                raise ValueError

        for nid, array_input in self.array_input_table.items():
            z3_expr = ts.put_input(
                array_input.sort.to_z3_sort(self.ctx), array_input.symbol
            )
            if isinstance(z3_expr, z3.ArrayRef):
                m.z3_array_m[nid] = z3_expr
            else:
                raise ValueError

        for nid, bitvec_state in self.bitvec_state_table.items():
            (m.z3_bitvec_or_bool_m[nid], m_post.z3_bitvec_or_bool_m[nid]) = ts.put_var(
                bitvec_state.sort.to_z3_sort(self.ctx), bitvec_state.symbol
            )

        for nid, array_state in self.array_state_table.items():
            (m.z3_array_m[nid], m_post.z3_array_m[nid]) = ts.put_var(
                array_state.sort.to_z3_sort(self.ctx), array_state.symbol
            )

        # noinspection DuplicatedCode
        for nid, bitvec_state in self.bitvec_state_table.items():
            if bitvec_state.init:
                z3_inits.append(
                    to_z3_eq(
                        bitvec_state.to_z3_expr(m), bitvec_state.init.to_z3_expr(m)
                    )
                )
            if bitvec_state.next:
                z3_nexts.append(
                    to_z3_eq(
                        bitvec_state.to_z3_expr(m_post), bitvec_state.next.to_z3_expr(m)
                    )
                )

        # noinspection DuplicatedCode
        for nid, array_state in self.array_state_table.items():
            if array_state.init:
                if array_state.sort.element_sort == array_state.init.sort:
                    z3_inits.append(
                        to_z3_eq(
                            array_state.to_z3_expr(m),
                            z3.K(
                                array_state.sort.index_sort.to_z3_sort(self.ctx),
                                try_bitvec_to_bool(array_state.init.to_z3_expr(m)),
                            ),
                        )
                    )
                else:
                    z3_inits.append(
                        to_z3_eq(
                            array_state.to_z3_expr(m), array_state.init.to_z3_expr(m)
                        )
                    )
            if array_state.next:
                z3_nexts.append(
                    to_z3_eq(
                        array_state.to_z3_expr(m_post), array_state.next.to_z3_expr(m)
                    )
                )

        ts.init = z3.And(z3_inits, self.ctx)
        ts.tr = z3.And(z3_nexts, self.ctx)
        ts.bads = [bad.bitvec.to_z3_bool(m) for bad in self.bad_list]
        ts.constraints = [
            constraint.bitvec.to_z3_bool(m) for constraint in self.constraint_list
        ]
        return ts


def btor2ts(
    input_file: TextIO, simplify: bool = True, recursion_limit: int = 10000
) -> Ts:
    if recursion_limit > 0:
        sys.setrecursionlimit(recursion_limit)

    btor2_parser: Btor2Parser = Btor2Parser(z3.Context())
    btor2_parser.parse(input_file)

    ts: Ts = btor2_parser.to_ts()
    ts.reduce_constraints()
    ts.reduce_bads()
    if simplify:
        ts.simplify()

    return ts


def btor2chc(
    input_file: TextIO,
    output_file: TextIO,
    simplify: bool = True,
    recursion_limit: int = 10000,
    engine: str = "spacer",
) -> None:
    ts: Ts = btor2ts(input_file, simplify, recursion_limit)

    if engine:
        output_file.write("(set-option :fp.engine {:s})\n".format(engine))

    output_file.writelines(generate_chc_str(*generate_vc(ts)))


def main() -> None:
    import argparse

    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="A tool to convert btor2 files to Constrained Horn Clauses (CHC) "
        "and pretty print them in the SMT-LIB format."
    )
    parser.add_argument(
        "input",
        metavar="FILE",
        help="input btor2 file",
        nargs="?",
        type=argparse.FileType("r"),
        default=sys.stdin,
    )
    parser.add_argument(
        "-output",
        "--output",
        metavar="FILE",
        help="place the output into [FILE]",
        nargs="?",
        type=argparse.FileType("w+"),
        default=sys.stdout,
    )

    args: argparse.Namespace = parser.parse_args()

    btor2chc(args.input, args.output)
    if args.input != sys.stdin:
        args.input.close()
    if args.output != sys.stdout:
        args.output.close()


if __name__ == "__main__":
    main()
