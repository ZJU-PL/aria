"""
GANSAT Custom GAN — Iterative Refinement GAN for SMT Solving

Novel architecture designed specifically for QF_LIA formula satisfying assignment prediction.

Core idea:
  Round 0 : Generator makes an initial assignment guess from formula encoding + noise
  Round 1+: ViolationComputer computes which constraints are violated and by how much
            RefinementStep adjusts the assignment targeted at violated constraints
  Repeat K times → progressively more valid assignment

This mirrors how a human solver works:
  "x=5, y=5 ... but x+y must be 7, so increase y and decrease x ... x=3, y=4 ✓"

Key modules:
  ViolationComputer     — differentiable constraint violation scoring
  InitialGuesser        — formula + noise → first assignment
  RefinementStep        — (formula, assignment, violations) → improved assignment
  IterativeGenerator    — chains InitialGuesser + K × RefinementStep
  Discriminator         — (formula, assignment) → real/fake logit
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoder import feature_dim, MAX_VARS, MAX_CONSTRAINTS, COEFF_CLIP, BOUND_CLIP

FORMULA_DIM     = feature_dim()       # 8576
ASSIGN_DIM      = MAX_VARS            # 64
NOISE_DIM       = 128
CONSTRAINT_DIM  = MAX_CONSTRAINTS     # 128  — per-constraint violation vector
N_ROUNDS        = 3                   # refinement rounds


# ─────────────────────────────────────────────
#  Shared building blocks
# ─────────────────────────────────────────────

class ResBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


# ─────────────────────────────────────────────
#  ViolationComputer
# ─────────────────────────────────────────────

class ViolationComputer(nn.Module):
    """
    Differentiably computes how much each constraint is violated
    by the current assignment.

    Formula encoding layout (from encoder.py):
      [0   : 128]         = variable bounds block  (64 vars × 2)
      [128 : 128+8448]    = constraint block        (128 constraints × 66)
                              each row: [coeff_0..63, rhs, ctype]

    All values are normalized to [-1, 1] in the encoding.
    We work in normalized space — gradients point in the correct direction.

    Returns:
      constraint_violations : [batch, 128]  — per-constraint violation score (>=0)
      variable_violations   : [batch, 64]   — per-variable responsibility score (>=0)
    """

    def forward(
        self,
        formula_enc: torch.Tensor,   # [batch, 8576]
        assignment:  torch.Tensor,   # [batch, 64]  values in [-1, 1]
    ):
        batch = formula_enc.size(0)

        # Extract constraint block
        constraint_block = formula_enc[:, 128:]                          # [batch, 8448]
        constraint_block = constraint_block.view(batch, MAX_CONSTRAINTS, MAX_VARS + 2)

        coeffs = constraint_block[:, :, :MAX_VARS]   # [batch, 128, 64]  normalized
        rhs    = constraint_block[:, :, MAX_VARS]     # [batch, 128]      normalized
        ctype  = constraint_block[:, :, MAX_VARS + 1] # [batch, 128]      0/1/2/3

        # Compute A·x in normalized space
        # assignment: [batch, 64] → [batch, 64, 1]
        Ax = torch.bmm(coeffs, assignment.unsqueeze(-1)).squeeze(-1)  # [batch, 128]

        diff = Ax - rhs  # positive → LHS exceeds RHS

        # Masks per constraint type (soft, based on stored float type index)
        # type encoding: 0=leq, 1=eq, 2=geq, 3=neq
        leq_mask = (ctype < 0.25).float()
        eq_mask  = ((ctype >= 0.25) & (ctype < 0.75)).float()
        geq_mask = ((ctype >= 0.75) & (ctype < 1.25)).float()

        v_leq = F.relu(diff)       * leq_mask   # violated if Ax > rhs
        v_eq  = diff.abs()         * eq_mask    # violated if Ax ≠ rhs
        v_geq = F.relu(-diff)      * geq_mask   # violated if Ax < rhs

        constraint_violations = v_leq + v_eq + v_geq  # [batch, 128]

        # Aggregate per variable: variables in highly-violated constraints get high score
        # [batch, 64] = violations [batch, 128] × |coeffs| [batch, 128, 64]
        variable_violations = torch.bmm(
            constraint_violations.unsqueeze(1),   # [batch, 1, 128]
            coeffs.abs()                           # [batch, 128, 64]
        ).squeeze(1)                               # [batch, 64]

        return constraint_violations, variable_violations


# ─────────────────────────────────────────────
#  Initial Guesser
# ─────────────────────────────────────────────

class InitialGuesser(nn.Module):
    """
    Produces an initial assignment from formula encoding + noise.
    No violation feedback yet — pure prior from training distribution.
    """

    def __init__(self, hidden: int = 512, depth: int = 4):
        super().__init__()
        in_dim = FORMULA_DIM + NOISE_DIM
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.LeakyReLU(0.2),
        )
        self.blocks  = nn.ModuleList([ResBlock(hidden) for _ in range(depth)])
        self.out     = nn.Sequential(nn.Linear(hidden, ASSIGN_DIM), nn.Tanh())

    def forward(self, formula_enc: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(torch.cat([formula_enc, noise], dim=-1))
        for block in self.blocks:
            x = block(x)
        return self.out(x)


# ─────────────────────────────────────────────
#  Refinement Step
# ─────────────────────────────────────────────

class RefinementStep(nn.Module):
    """
    One round of violation-guided refinement.

    Input:
      formula_enc          [batch, 8576]
      current_assignment   [batch, 64]
      constraint_violations [batch, 128]
      variable_violations  [batch, 64]

    Output:
      refined_assignment   [batch, 64]  — small delta added to current (residual)
    """

    def __init__(self, hidden: int = 256):
        super().__init__()
        in_dim = FORMULA_DIM + ASSIGN_DIM + CONSTRAINT_DIM + ASSIGN_DIM
        #        8576          64            128               64         = 8832

        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.LeakyReLU(0.2),
        )
        self.blocks = nn.ModuleList([ResBlock(hidden) for _ in range(3)])

        # Predict a delta (adjustment), not an absolute value
        self.delta_head = nn.Sequential(
            nn.Linear(hidden, ASSIGN_DIM),
            nn.Tanh(),
        )
        # Learned step size per variable (how aggressive the adjustment is)
        self.step_scale = nn.Parameter(torch.ones(ASSIGN_DIM) * 0.1)

    def forward(
        self,
        formula_enc:           torch.Tensor,
        current_assignment:    torch.Tensor,
        constraint_violations: torch.Tensor,
        variable_violations:   torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat(
            [formula_enc, current_assignment, constraint_violations, variable_violations],
            dim=-1
        )
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)

        delta = self.delta_head(x) * self.step_scale  # small targeted adjustment
        refined = (current_assignment + delta).clamp(-1.0, 1.0)
        return refined


# ─────────────────────────────────────────────
#  Iterative Generator (full)
# ─────────────────────────────────────────────

class IterativeGenerator(nn.Module):
    """
    Full GAN Generator with iterative violation-guided refinement.

    Forward pass:
      1. InitialGuesser → first assignment (round 0)
      2. For each round 1..N_ROUNDS:
           a. ViolationComputer scores current assignment
           b. RefinementStep adjusts assignment toward constraint satisfaction
      3. Return final assignment

    The refinement steps share weights (parameter efficient, generalizes better).
    """

    def __init__(self, n_rounds: int = N_ROUNDS):
        super().__init__()
        self.n_rounds          = n_rounds
        self.initial_guesser   = InitialGuesser(hidden=512, depth=4)
        self.violation_computer = ViolationComputer()
        self.refine            = RefinementStep(hidden=256)  # shared across all rounds

    def forward(
        self,
        formula_enc: torch.Tensor,
        noise:       torch.Tensor = None,
        return_trajectory: bool = False,
    ) -> torch.Tensor:
        if noise is None:
            noise = torch.randn(formula_enc.size(0), NOISE_DIM, device=formula_enc.device)

        # Round 0: initial guess
        assignment = self.initial_guesser(formula_enc, noise)
        trajectory = [assignment] if return_trajectory else None

        # Rounds 1..K: iterative refinement
        for _ in range(self.n_rounds):
            c_viol, v_viol = self.violation_computer(formula_enc, assignment)
            assignment = self.refine(formula_enc, assignment, c_viol, v_viol)
            if return_trajectory:
                trajectory.append(assignment)

        if return_trajectory:
            return assignment, trajectory
        return assignment

    def sample(self, formula_enc: torch.Tensor, n_samples: int = 1) -> torch.Tensor:
        """Generate n_samples diverse candidates per formula (different noise seeds)."""
        B = formula_enc.size(0)
        expanded = formula_enc.unsqueeze(1).expand(B, n_samples, -1).reshape(B * n_samples, -1)
        noise    = torch.randn(B * n_samples, NOISE_DIM, device=formula_enc.device)
        out      = self.forward(expanded, noise)
        return out.view(B, n_samples, ASSIGN_DIM)

    def violation_score(self, formula_enc: torch.Tensor, assignment: torch.Tensor) -> torch.Tensor:
        """Total violation score for an assignment — lower is better."""
        c_viol, _ = self.violation_computer(formula_enc, assignment)
        return c_viol.sum(dim=-1)  # [batch]


# ─────────────────────────────────────────────
#  Discriminator
# ─────────────────────────────────────────────

class Discriminator(nn.Module):
    """
    Scores (formula, assignment) pairs as real (satisfying) or fake.

    Also receives the violation signal to make it violation-aware —
    it learns that real pairs have near-zero violation.
    """

    def __init__(self, hidden: int = 512, depth: int = 4):
        super().__init__()
        self.violation_computer = ViolationComputer()

        # Input: formula + assignment + constraint_violations + variable_violations
        in_dim = FORMULA_DIM + ASSIGN_DIM + CONSTRAINT_DIM + ASSIGN_DIM
        #        8576          64            128               64          = 8832

        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.LeakyReLU(0.2),
        )
        self.blocks = nn.ModuleList([ResBlock(hidden) for _ in range(depth)])
        self.output  = nn.Linear(hidden, 1)

    def forward(self, formula_enc: torch.Tensor, assignment: torch.Tensor) -> torch.Tensor:
        c_viol, v_viol = self.violation_computer(formula_enc, assignment)
        x = torch.cat([formula_enc, assignment, c_viol, v_viol], dim=-1)
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        return self.output(x).squeeze(-1)


# ─────────────────────────────────────────────
#  Utility
# ─────────────────────────────────────────────

def assignment_to_tensor(assignment_dict: dict, var_names: list) -> torch.Tensor:
    """Convert {var_name: int_value} to normalized [-1, 1] tensor."""
    vec = torch.zeros(MAX_VARS)
    for i, name in enumerate(var_names[:MAX_VARS]):
        val = assignment_dict.get(name, 0)
        vec[i] = float(val) / BOUND_CLIP
    return vec.clamp(-1.0, 1.0)


# Keep backward-compatible alias
Generator = IterativeGenerator
