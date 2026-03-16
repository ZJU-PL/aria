#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Affine relation analysis over GF(2^k), powered by `galois`.

核心思想：
1. 选定域 GF(2^k)（需给出不可约多项式）。
2. 所有矩阵元素都用 galois.GFArray 对象存储。
3. 行最简化 (Gauss–Jordan) 需使用乘法/逆元，而非简单 XOR。
4. 高层操作（meet / join / compose / star）逻辑与 GF(2) 版完全一致。
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import numpy as np
import galois


# ----------------------------------------------------------------------------
# 1. 选择域 GF(2^k)
# ----------------------------------------------------------------------------
k = 4  # 自行修改
GF = galois.GF(2**k, irreducible_poly=galois.conway_poly(2, k))  # 或者  # 指定多项式
ZERO = GF(0)
ONE = GF(1)


# ----------------------------------------------------------------------------
# 2. 低层工具
# ----------------------------------------------------------------------------
def row_reduce(mat: np.ndarray) -> np.ndarray:
    """
    Gauss–Jordan elimination over GF(2^k).
    `mat` 会被就地修改；返回它自身（行最简形）。
    """
    m, num_cols = mat.shape
    r = 0  # 当前 pivot 行
    for c in range(num_cols - 1):  # 最后一列照样参与
        # 1) 找到非零 pivot
        pivot = None
        for i in range(r, m):
            if mat[i, c] != ZERO:
                pivot = i
                break
        if pivot is None:
            continue

        # 2) 行交换
        if pivot != r:
            mat[[r, pivot]] = mat[[pivot, r]]

        # 3) 归一化
        inv = mat[r, c] ** -1  # galois 数组可直接求逆
        mat[r, :] *= inv

        # 4) 消元
        for i in range(m):
            if i != r and mat[i, c] != ZERO:
                coeff = mat[i, c]
                mat[i, :] -= coeff * mat[r, :]

        r += 1
        if r == m:
            break
    return mat


def stack(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Stack two matrices vertically; they must have the same width."""
    if a.shape[1] != b.shape[1]:
        raise ValueError("stack: incompatible widths")
    return np.vstack((a, b))


def project_cols(mat: np.ndarray, rng: Tuple[int, int]) -> np.ndarray:
    """
    ∃-quantifier / 删列（与 GF(2) 版相同，但注意用 all/any 判断零行）
    rng = (j, i) 表示删去 [i, j) 列。
    若一行在删去区间含非零，则该行可被满足 ⇒ 删除整行。
    """
    j, i = rng
    if not 0 <= i <= j <= mat.shape[1] - 1:
        raise ValueError("invalid range")

    seg = mat[:, i:j]
    keep_mask = np.all(seg == ZERO, axis=1)  # 全零才保留
    kept = mat[keep_mask]

    cols = list(range(0, i)) + list(range(j, mat.shape[1]))
    return kept[:, cols]


# ----------------------------------------------------------------------------
# 3. 抽象域
# ----------------------------------------------------------------------------
@dataclass
class AffineRelation:
    """Affine relation over GF(2^k) represented as a matrix."""

    mat: np.ndarray  # dtype = GF

    # ---------- 基本信息 ----------
    @property
    def m(self) -> int:
        """Number of constraint rows."""
        return self.mat.shape[0]

    @property
    def n(self) -> int:  # 变量数
        """Number of variables."""
        return self.mat.shape[1] - 1

    # ---------- 人类可读 ----------
    def __str__(self) -> str:
        if self.m == 0:
            return "⊤"
        rows = []
        for r in self.mat:
            coeffs = " ".join(f"{int(x):2d}" for x in r[:-1])
            rows.append(f"[{coeffs} | {int(r[-1])}]")
        return "\n".join(rows)

    # ---------- 构造 ----------
    @staticmethod
    def top(n: int) -> "AffineRelation":
        """Create top element (no constraints)."""
        return AffineRelation(GF.Zeros((0, n + 1)))

    @staticmethod
    def bot(n: int) -> "AffineRelation":
        """Create bottom element (unsatisfiable)."""
        row = GF.Zeros((1, n + 1))
        row[0, -1] = ONE
        return AffineRelation(row)

    @staticmethod
    def eye(n: int) -> "AffineRelation":
        """Create identity relation x' = x."""
        mat = GF.Zeros((n, n + 1))
        for i in range(n):
            mat[i, i] = ONE
        return AffineRelation(mat)

    # ---------- 规范化 / 可满足 ----------
    def canonicalize(self) -> "AffineRelation":
        """Canonicalize the relation by row reduction."""
        mat = row_reduce(self.mat.copy())
        # 如果出现 [0 … 0 | 1] ⇒ ⊥
        unsat = np.all(mat[:, :-1] == ZERO, axis=1) & (mat[:, -1] != ZERO)
        if unsat.any():
            return AffineRelation.bot(self.n)
        return AffineRelation(mat)

    # ---------- meet / join ----------
    def meet(self, other: "AffineRelation") -> "AffineRelation":
        """Compute the meet (intersection) of two relations."""
        if self.m == 0:
            return other
        if other.m == 0:
            return self
        if self.n != other.n:
            raise ValueError("meet: different variable sets")
        return AffineRelation(stack(self.mat, other.mat)).canonicalize()

    def join(self, other: "AffineRelation") -> "AffineRelation":
        """Compute the join (union) of two relations."""
        if self.m == 0:
            return self
        if other.m == 0:
            return other
        if self.n != other.n:
            raise ValueError("join: different variable sets")

        num_vars = self.n
        top_block = np.hstack((self.mat, self.mat))

        # OLD: bot_block = np.hstack((GF.Zeros_like(other.mat), other.mat))
        zeros_like_other = GF.Zeros(other.mat.shape)
        bot_block = np.hstack((zeros_like_other, other.mat))

        tmp = np.vstack((top_block, bot_block))
        prj = project_cols(tmp, (num_vars + 1, 0))
        return AffineRelation(prj).canonicalize()

    # ---------- 顺序合成 ----------
    def compose(self, other: "AffineRelation") -> "AffineRelation":
        """Compose two relations: self ∘ other."""
        if self.n != other.n:
            raise ValueError("compose: variable sets mismatch")
        num_vars = self.n
        if num_vars % 2:
            raise ValueError("compose: n must be even (X'|X)")

        v = num_vars // 2
        # other: [c | 0^v | A_x]
        c2 = other.mat[:, -1:]
        zeros2 = GF.Zeros((other.m, v))
        ax2 = other.mat[:, : 2 * v]
        rows2 = np.hstack((c2, zeros2, ax2))
        # self:  [A | c | 0^v]
        a1c1 = self.mat
        zeros1 = GF.Zeros((self.m, v))
        rows1 = np.hstack((a1c1, zeros1))

        tmp = np.vstack((rows2, rows1))
        prj = project_cols(tmp, (2 * v, v))
        return AffineRelation(prj).canonicalize()

    # ---------- Kleene 星 ----------
    def star(self) -> "AffineRelation":
        """Compute the Kleene star (transitive closure)."""
        cur = AffineRelation.eye(self.n // 2 * 2)  # 保证偶数
        nxt = self.join(cur.compose(self))
        while not np.array_equal(nxt.mat, cur.mat):
            cur = nxt
            nxt = self.join(cur.compose(self))
        return nxt

    # ---------- 其他 ----------
    def copy(self) -> "AffineRelation":
        """Create a copy of this relation."""
        return AffineRelation(self.mat.copy())

    def __eq__(self, other: object) -> bool:
        """Check equality of two relations."""
        if not isinstance(other, AffineRelation):
            return False
        return np.array_equal(self.canonicalize().mat, other.canonicalize().mat)


# ----------------------------------------------------------------------------
# 4. DEMO
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Using field: {GF};  primitive polynomial = {GF.irreducible_poly}")

    N = 4  # 变量个数
    r1 = AffineRelation.eye(N)  # x' = x
    r2 = r1.copy()
    r2.mat[0, -1] = ONE  # 翻转第 0 位 (x'0 += 1)

    print("\nr1 (identity):")
    print(r1)
    print("\nr2 (negate bit 0):")
    print(r2)

    print("\nmeet(r1, r2):")
    print(r1.meet(r2))

    print("\njoin(r1, r2):")
    print(r1.join(r2))

    # 组合示例（要求 n = 2v）
    V = 2
    inc = AffineRelation.eye(V * 2)  # [X'|X] 共 4 列
    inc.mat[0] += inc.mat[1]  # x'0 += x'1

    swap = AffineRelation.eye(V * 2)
    swap.mat[[0, 1], :V] = swap.mat[[1, 0], :V]

    comp = inc.compose(swap)
    print("\ncompose(inc, swap):")
    print(comp)
