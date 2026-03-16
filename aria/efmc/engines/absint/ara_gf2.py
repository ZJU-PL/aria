#!/usr/bin/env python3
"""
Affine relation analysis over GF(2)
translated from https://github.com/meamy/feynman/blob/master/src/Feynman/Algebra/AffineRel.hs
(to check)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

import numpy as np  # 0/1 matrices, XOR → addition modulo-2


# ---------------------------------------------------------------------------
# 低层工具
# ---------------------------------------------------------------------------


def row_reduce(mat: np.ndarray) -> np.ndarray:
    """
    Gauss–Jordan elimination over GF(2), 返回行最简（RREF）矩阵。
    mat 只能包含 0/1，shape = (m, n)；该函数就地修改并返回引用。
    """
    m, num_cols = mat.shape
    r = 0  # 当前 pivot 行
    for c in range(num_cols - 1):  # 最后一列是常数项，也可参与
        # 1) 选 pivot
        pivot = None
        for i in range(r, m):
            if mat[i, c]:
                pivot = i
                break
        if pivot is None:
            continue

        # 2) 行交换
        if pivot != r:
            mat[[r, pivot]] = mat[[pivot, r]]

        # 3) 消元
        for i in range(m):
            if i != r and mat[i, c]:
                mat[i] ^= mat[r]  # GF(2) 下减法就是 XOR

        r += 1
        if r == m:
            break

    return mat


def stack(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    "把两矩阵竖直拼接；列数必须一致"
    if a.shape[1] != b.shape[1]:
        raise ValueError("stack: incompatible widths")
    return np.vstack((a, b))


def project_cols(mat: np.ndarray, rng: Tuple[int, int]) -> np.ndarray:
    """
    存在量化地“删列”。等价于 Elder-KS 里的 ⟨∃x.φ⟩：
    rng = (j, i) 表示删去列 i..j-1 （左闭右开）。
    若一行在被删区间里含 1，则说明它可以被满足 → 整行丢弃。
    其余行保留，并把对应列删除。
    """
    j, i = rng
    if not 0 <= i <= j <= mat.shape[1] - 1:
        raise ValueError("invalid range")

    keep_mask = mat[:, i:j].sum(axis=1) == 0  # 那段全 0 才留
    kept = mat[keep_mask]

    # 实际删除列
    cols = list(range(0, i)) + list(range(j, mat.shape[1]))
    return kept[:, cols]


# ---------------------------------------------------------------------------
# 抽象域数据结构
# ---------------------------------------------------------------------------


@dataclass
class AffineRelation:
    """
    行向量 r = [a₀ … aₙ-₁ | c]  代表方程  ∑ aᵢ·xᵢ  =  c   (mod 2)
    self.mat.shape = (m, n+1)
    """

    mat: np.ndarray  # dtype uint8 / bool

    # ------------------------- 基本信息 -------------------------

    @property
    def m(self) -> int:  # 约束行数
        """Number of constraint rows."""
        return self.mat.shape[0]

    @property
    def n(self) -> int:  # 变量个数
        """Number of variables."""
        return self.mat.shape[1] - 1

    # ------------------------- 可读输出 -------------------------

    def __str__(self) -> str:
        if self.m == 0:
            return "⊤"  # top
        rows = []
        for r in self.mat:
            left = "".join(str(int(b)) for b in r[:-1])
            rows.append(f"[{left} | {int(r[-1])}]")
        return "\n".join(rows)

    # ------------------------- 构造子 -------------------------

    @staticmethod
    def top(n: int) -> "AffineRelation":
        "空矩阵 → 没有约束"
        return AffineRelation(np.empty((0, n + 1), dtype=np.uint8))

    @staticmethod
    def bot(n: int) -> "AffineRelation":
        "0 = 1 → 不可满足"
        row = np.zeros((1, n + 1), dtype=np.uint8)
        row[0, -1] = 1
        return AffineRelation(row)

    @staticmethod
    def eye(n: int) -> "AffineRelation":
        "恒等关系  x' = x   （这里仍然按 [X'|X|c] 的缩减存储）"
        mat = np.zeros((n, n + 1), dtype=np.uint8)
        for i in range(n):
            mat[i, i] = 1
        return AffineRelation(mat)

    # ------------------------- Canonicalize / SAT 检查 -------------------------

    def canonicalize(self) -> "AffineRelation":
        """Canonicalize the relation by row reduction."""
        mat = row_reduce(self.mat.copy())

        # 若出现 [0 … 0 | 1] 说明约束组 UNSAT → ⊥
        unsat = np.all(mat[:, :-1] == 0, axis=1) & (mat[:, -1] == 1)
        if unsat.any():
            return AffineRelation.bot(self.n)
        return AffineRelation(mat)

    # ------------------------- 小工具 -------------------------

    def add_vars(self, k: int) -> "AffineRelation":
        """在末尾再引入 k 个自由变量；添上恒等行"""
        m, n1 = self.mat.shape  # n1 = n+1
        num_vars = n1 - 1
        new_n = num_vars + k
        new_mat = np.zeros((m + k, new_n + 1), dtype=np.uint8)

        # 复制旧列
        new_mat[:m, :num_vars] = self.mat[:, :num_vars]
        new_mat[:m, -1] = self.mat[:, -1]

        # 加上 k 行单位
        for i in range(k):
            new_mat[m + i, num_vars + i] = 1

        return AffineRelation(new_mat)

    def negate_post(self, j: int) -> "AffineRelation":
        """x'j ← x'j + 1"""
        mat = self.mat.copy()
        mat[j, -1] ^= 1
        return AffineRelation(mat)

    def add_post(self, j: int, k: int) -> "AffineRelation":
        """x'j ← x'j + x'k"""
        mat = self.mat.copy()
        mat[j] ^= mat[k]
        return AffineRelation(mat)

    def clear_post(self, j: int) -> "AffineRelation":
        """x'j ← 0"""
        mat = self.mat.copy()
        mat[j, :] = 0
        return AffineRelation(mat)

    # ------------------------- 交 (meet) / 并 (join) -------------------------

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
        """
        Nelson–Oppen trick：加入一个布尔 guard 列，随后存在量化消去。
        """
        if self.m == 0:  # ⊤ ∪ anything = ⊤
            return self
        if other.m == 0:
            return other
        if self.n != other.n:
            raise ValueError("join: different variable sets")

        v = self.n
        # [ r | r ]  /  [0 | r]
        top_block = np.hstack((self.mat[:, :], self.mat[:, :]))
        bot_block = np.hstack((np.zeros_like(other.mat), other.mat))
        tmp = np.vstack((top_block, bot_block))

        # 投影掉左 v+1 列（include 常数） ==> [X|c] 左边那份存在量化
        prj = project_cols(tmp, (v + 1, 0))
        return AffineRelation(prj).canonicalize()

    # ------------------------- 顺序合成 compose -------------------------

    # ------------------------- 顺序合成 compose -------------------------

    def compose(self, other: "AffineRelation") -> "AffineRelation":
        """
        SELF ∘ OTHER ：先执行 other，然后执行 self
        变量排列必须都是  [X' | X]，即 n = 2·v（偶数）
        """
        if self.n != other.n:
            raise ValueError("compose: variable sets not equal")

        num_vars = self.n
        if num_vars % 2:
            raise ValueError("compose: expect even number of vars (X',X)")

        v = num_vars // 2

        # ---------- 生成 ar2 的行 :  [ c | 0^v | A_x ]
        c2 = other.mat[:, -1:]  # (m2,1)
        zeros2 = np.zeros((other.m, v), dtype=np.uint8)
        ax2 = other.mat[:, : 2 * v]  # 去掉常数列
        rows2 = np.hstack((c2, zeros2, ax2))  # (m2, 3v+1)

        # ---------- 生成 ar1 的行 :  [ A | c | 0^v ]
        a1c1 = self.mat  # (m1, 2v+1)
        zeros1 = np.zeros((self.m, v), dtype=np.uint8)
        rows1 = np.hstack((a1c1, zeros1))  # (m1, 3v+1)

        tmp = np.vstack((rows2, rows1))  # (m1+m2, 3v+1)

        # 删掉中间那 v 列（索引 v .. 2v-1）
        prj = project_cols(tmp, (2 * v, v))  # 返回仍是 ( ?, 2v+1 )

        return AffineRelation(prj).canonicalize()

    # ------------------------- Kleene star -------------------------

    def star(self) -> "AffineRelation":
        """
        R* = μX.(I ∪ R∘X)    —— 迭代到不动点即可
        （这里用最朴素的 worklist 算法）
        """
        cur = AffineRelation.eye(self.n // 2 * 2)  # 保证偶数
        nxt = self.join(cur.compose(self))
        while str(nxt.mat.tobytes()) != str(cur.mat.tobytes()):
            cur = nxt
            nxt = self.join(cur.compose(self))
        return nxt

    # ------------------------- 方便的等价 / 拷贝 -------------------------

    def copy(self) -> "AffineRelation":
        """Create a copy of this relation."""
        return AffineRelation(self.mat.copy())

    def __eq__(self, other: object) -> bool:
        """Check equality of two relations."""
        if not isinstance(other, AffineRelation):
            return False
        return np.array_equal(self.canonicalize().mat, other.canonicalize().mat)


# ---------------------------------------------------------------------------
# DEMO / quick self-check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    N = 4
    r1 = AffineRelation.eye(N)  # x' = x
    r2 = r1.negate_post(0)  # flip 第 0 位
    r3 = r2.add_post(1, 0)  # x'1 += x'0

    print("r1 (identity):\n", r1, sep="")
    print("\nr2 (negate bit 0):\n", r2, sep="")
    print("\nr3 (also add bit0 -> bit1):\n", r3, sep="")

    # meet / join
    print("\nmeet(r2, r3):\n", r2.meet(r3), sep="")
    print("\njoin(r2, r3):\n", r2.join(r3), sep="")

    # composition   (这里把变量看成两半 [X'|X] → 需偶数)
    N2 = 2
    inc = AffineRelation.eye(N2)  # 恒等
    inc = inc.add_post(0, 1)  # x'0 += x1
    swap = AffineRelation.eye(N2)
    swap.mat[[0, 1], :N2] = swap.mat[[1, 0], :N2]  # 交换 x0 x1

    composed = inc.compose(swap)
    print("\ncompose(inc, swap):\n", composed, sep="")
