"""
QF_BV GAN — Iterative Refinement GAN for bit-vector theory.

Extends the QF_LIA design with a BV-specific ViolationComputer that
handles unsigned/signed comparisons and bitwise operations numerically.

Key differences from QF_LIA GAN:
  - Formula encoding dim: 10752 (vs 8576)
  - Assignment values in [-1,1] map to [0, 2^width - 1] per variable
  - Violation computed with masking to bit-width boundaries
  - Bitwise ops approximated: AND ≈ min, OR ≈ max, XOR ≈ |a-b|
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .bv_encoder import bv_feature_dim, MAX_VARS, MAX_CONSTRAINTS, OP_DIM, VAR_FEAT

BV_FORMULA_DIM = bv_feature_dim()   # 10752
BV_ASSIGN_DIM  = MAX_VARS           # 64
BV_NOISE_DIM   = 128
BV_N_ROUNDS    = 3


# ─── Shared ResBlock (same as LIA GAN) ────────────────────────────────────────

class BVResBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim), nn.LayerNorm(dim), nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(dim, dim), nn.LayerNorm(dim),
        )
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        return self.act(x + self.net(x))


# ─── BV Violation Computer ────────────────────────────────────────────────────

class BVViolationComputer(nn.Module):
    """
    Differentiable violation scoring for QF_BV constraints.

    Formula encoding layout (from bv_encoder.py):
      [0        : MAX_VARS*VAR_FEAT]       variable block  256 dims
      [256      : 256 + MAX_CONSTRAINTS*(MAX_VARS+OP_DIM+2)]  constraint block

    Each constraint row: [var_mask(64), op_onehot(16), lhs_val(1), rhs_val(1)]

    Violation approximations:
      EQ  (op 8) : |a_norm - b_norm|
      ULT (op 0) : ReLU(a_norm - b_norm)     (violated if a >= b)
      ULE (op 1) : ReLU(a_norm - b_norm - ε)
      UGT (op 2) : ReLU(b_norm - a_norm)
      UGE (op 3) : ReLU(b_norm - a_norm - ε)
      SLT-SGE    : same as unsigned (signed handled by GAN learning)
      BVAND (10) : ReLU(-(a·b))  — violated if AND is nonzero
      BVOR  (11) : ReLU(1 - (a+b))  — violated if OR is zero
      BVXOR (12) : |a - b| flipped
      ARITH(13+) : |a ± b - expected|
    """

    def forward(self, formula_enc: torch.Tensor, assignment: torch.Tensor):
        B = formula_enc.size(0)
        var_block_size = MAX_VARS * VAR_FEAT  # 256
        row_size = MAX_VARS + OP_DIM + 2       # 82

        constraint_block = formula_enc[:, var_block_size:]        # [B, 10496]
        constraint_block = constraint_block.view(B, MAX_CONSTRAINTS, row_size)

        var_mask  = constraint_block[:, :, :MAX_VARS]              # [B,128,64]
        op_onehot = constraint_block[:, :, MAX_VARS:MAX_VARS+OP_DIM] # [B,128,16]
        lhs_val   = constraint_block[:, :, MAX_VARS+OP_DIM]        # [B,128]
        rhs_val   = constraint_block[:, :, MAX_VARS+OP_DIM+1]      # [B,128]

        # Effective LHS/RHS: weighted sum of assignment values by var_mask
        # assignment: [B, 64] → [B, 1, 64] × var_mask [B, 128, 64]
        a_eff = (assignment.unsqueeze(1) * var_mask).sum(dim=-1)   # [B, 128]
        b_eff = rhs_val                                             # [B, 128]
        diff  = a_eff - b_eff

        eps = 1e-4
        # Per-op violations (multiply by op mask to select correct formula)
        v = torch.zeros_like(diff)

        v += F.relu(diff)         * op_onehot[:,:,0]   # ULT: a >= b → violated
        v += F.relu(diff - eps)   * op_onehot[:,:,1]   # ULE: a > b
        v += F.relu(-diff)        * op_onehot[:,:,2]   # UGT: a <= b
        v += F.relu(-diff - eps)  * op_onehot[:,:,3]   # UGE: a < b
        v += F.relu(diff)         * op_onehot[:,:,4]   # SLT (approx unsigned)
        v += F.relu(diff - eps)   * op_onehot[:,:,5]   # SLE
        v += F.relu(-diff)        * op_onehot[:,:,6]   # SGT
        v += F.relu(-diff - eps)  * op_onehot[:,:,7]   # SGE
        v += diff.abs()           * op_onehot[:,:,8]   # EQ
        v += F.relu(eps - diff.abs()) * op_onehot[:,:,9]  # DISTINCT: violated if equal
        # Bitwise approx
        v += F.relu(-(a_eff * b_eff))     * op_onehot[:,:,10]  # AND ≈ 0 if product > 0
        v += F.relu(1-(a_eff + b_eff))    * op_onehot[:,:,11]  # OR
        v += (a_eff - b_eff).abs()        * op_onehot[:,:,12]  # XOR ≈ difference
        v += (a_eff + b_eff - lhs_val).abs() * op_onehot[:,:,13]  # ADD
        v += (a_eff - b_eff - lhs_val).abs() * op_onehot[:,:,14]  # SUB
        v += (a_eff * b_eff - lhs_val).abs() * op_onehot[:,:,15]  # MUL

        constraint_violations = v   # [B, 128]

        # Per-variable responsibility
        variable_violations = torch.bmm(
            constraint_violations.unsqueeze(1),   # [B, 1, 128]
            var_mask.abs()                         # [B, 128, 64]
        ).squeeze(1)                               # [B, 64]

        return constraint_violations, variable_violations


# ─── BV InitialGuesser ────────────────────────────────────────────────────────

class BVInitialGuesser(nn.Module):
    def __init__(self, hidden: int = 512, depth: int = 4):
        super().__init__()
        in_dim = BV_FORMULA_DIM + BV_NOISE_DIM
        self.proj  = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.LeakyReLU(0.2)
        )
        self.blocks = nn.ModuleList([BVResBlock(hidden) for _ in range(depth)])
        self.out    = nn.Sequential(nn.Linear(hidden, BV_ASSIGN_DIM), nn.Tanh())

    def forward(self, f, z):
        x = self.proj(torch.cat([f, z], dim=-1))
        for b in self.blocks:
            x = b(x)
        return self.out(x)


# ─── BV RefinementStep ───────────────────────────────────────────────────────

class BVRefinementStep(nn.Module):
    def __init__(self, hidden: int = 256):
        super().__init__()
        in_dim = BV_FORMULA_DIM + BV_ASSIGN_DIM + MAX_CONSTRAINTS + BV_ASSIGN_DIM
        self.proj   = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.LeakyReLU(0.2)
        )
        self.blocks     = nn.ModuleList([BVResBlock(hidden) for _ in range(3)])
        self.delta_head = nn.Sequential(nn.Linear(hidden, BV_ASSIGN_DIM), nn.Tanh())
        self.step_scale = nn.Parameter(torch.ones(BV_ASSIGN_DIM) * 0.1)

    def forward(self, f, x, cv, vv):
        h = self.proj(torch.cat([f, x, cv, vv], dim=-1))
        for b in self.blocks:
            h = b(h)
        delta = self.delta_head(h) * self.step_scale
        return (x + delta).clamp(-1.0, 1.0)


# ─── BV Iterative Generator ───────────────────────────────────────────────────

class BVIterativeGenerator(nn.Module):
    """
    Full GAN Generator for QF_BV — identical pipeline to QF_LIA but
    uses BVViolationComputer and BV-specific dimensions.
    """

    def __init__(self, n_rounds: int = BV_N_ROUNDS):
        super().__init__()
        self.n_rounds  = n_rounds
        self.guesser   = BVInitialGuesser()
        self.vc        = BVViolationComputer()
        self.refine    = BVRefinementStep()

    def forward(self, f, z=None, return_trajectory=False):
        if z is None:
            z = torch.randn(f.size(0), BV_NOISE_DIM, device=f.device)
        x    = self.guesser(f, z)
        traj = [x] if return_trajectory else None

        for _ in range(self.n_rounds):
            cv, vv = self.vc(f, x)
            x = self.refine(f, x, cv, vv)
            if return_trajectory:
                traj.append(x)

        return (x, traj) if return_trajectory else x

    def sample(self, f, n_samples=1):
        B = f.size(0)
        fe = f.unsqueeze(1).expand(B, n_samples, -1).reshape(B * n_samples, -1)
        z  = torch.randn(B * n_samples, BV_NOISE_DIM, device=f.device)
        return self.forward(fe, z).view(B, n_samples, BV_ASSIGN_DIM)

    def violation_score(self, f, x):
        cv, _ = self.vc(f, x)
        return cv.sum(dim=-1)


# ─── BV Discriminator ────────────────────────────────────────────────────────

class BVDiscriminator(nn.Module):
    def __init__(self, hidden: int = 512, depth: int = 4):
        super().__init__()
        self.vc = BVViolationComputer()
        in_dim  = BV_FORMULA_DIM + BV_ASSIGN_DIM + MAX_CONSTRAINTS + BV_ASSIGN_DIM
        self.proj   = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.LeakyReLU(0.2)
        )
        self.blocks = nn.ModuleList([BVResBlock(hidden) for _ in range(depth)])
        self.out    = nn.Linear(hidden, 1)

    def forward(self, f, x):
        cv, vv = self.vc(f, x)
        h = self.proj(torch.cat([f, x, cv, vv], dim=-1))
        for b in self.blocks:
            h = b(h)
        return self.out(h).squeeze(-1)


# ─── Utility ─────────────────────────────────────────────────────────────────

def bv_assignment_to_tensor(assignment_dict: dict, var_names: list,
                             widths: dict) -> torch.Tensor:
    """Convert {var: int_value} to normalized [-1,1] tensor for BV vars."""
    vec = torch.zeros(MAX_VARS)
    for i, name in enumerate(var_names[:MAX_VARS]):
        val   = assignment_dict.get(name, 0)
        width = widths.get(name, 32)
        maxv  = float((1 << width) - 1)
        vec[i] = (2.0 * val / maxv - 1.0) if maxv > 0 else 0.0
    return vec.clamp(-1.0, 1.0)
