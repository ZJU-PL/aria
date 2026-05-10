#!/usr/bin/env bash
# ── setup_fstar.sh ────────────────────────────────────────────────────
# Build F*, Pulse, and KaRaMeL from source (fstar2 branch).
#
# Usage:
#   ./setup_fstar.sh <target_dir>
#
# After completion, the toolchain is at:
#   <target_dir>/bin/fstar.exe
#   <target_dir>/karamel/krml
#
# Environment:
#   FSTAR_REPO     Git URL (default: https://github.com/FStarLang/FStar.git)
#   FSTAR_BRANCH   Branch  (default: fstar2)
#   JOBS           Parallelism (default: nproc)
#   SKIP_BUILD     Set to 1 to skip build if already present
# ──────────────────────────────────────────────────────────────────────
set -euo pipefail

FSTAR_REPO="${FSTAR_REPO:-https://github.com/FStarLang/FStar.git}"
FSTAR_BRANCH="${FSTAR_BRANCH:-fstar2}"
JOBS="${JOBS:-$(nproc 2>/dev/null || echo 4)}"
SKIP_BUILD="${SKIP_BUILD:-0}"

TARGET_DIR="${1:?Usage: $0 <target_dir>}"
# Canonicalize: ensure parent exists so we can resolve the absolute path
mkdir -p "$(dirname "$TARGET_DIR")"
TARGET_DIR="$(cd "$(dirname "$TARGET_DIR")" && pwd)/$(basename "$TARGET_DIR")"

# ── Quick check: already built? ──────────────────────────────────────
if [ "$SKIP_BUILD" = "1" ] && [ -x "$TARGET_DIR/bin/fstar.exe" ]; then
    echo "F* already built at $TARGET_DIR/bin/fstar.exe — skipping (SKIP_BUILD=1)"
    "$TARGET_DIR/bin/fstar.exe" --version
    exit 0
fi

# ── Clone or update ──────────────────────────────────────────────────
echo "═══ Setting up F* ($FSTAR_BRANCH) in $TARGET_DIR ═══"
if [ -d "$TARGET_DIR/.git" ]; then
    echo "Updating existing checkout..."
    git -C "$TARGET_DIR" fetch origin "$FSTAR_BRANCH"
    git -C "$TARGET_DIR" checkout "$FSTAR_BRANCH"
    git -C "$TARGET_DIR" pull --ff-only
else
    echo "Cloning $FSTAR_REPO @ $FSTAR_BRANCH..."
    mkdir -p "$(dirname "$TARGET_DIR")"
    git clone --branch "$FSTAR_BRANCH" "$FSTAR_REPO" "$TARGET_DIR"
fi

cd "$TARGET_DIR"
git submodule update --init karamel

# ── Install Z3 ───────────────────────────────────────────────────────
if ! command -v z3 &>/dev/null && ! command -v z3-4.13.3 &>/dev/null; then
    echo "Installing Z3..."
    Z3_DIR="${Z3_DIR:-$TARGET_DIR/.z3}"
    mkdir -p "$Z3_DIR"
    bash .scripts/get_fstar_z3.sh "$Z3_DIR"
    export PATH="$Z3_DIR:$PATH"
    echo "Z3 installed to $Z3_DIR"
fi

# ── Install OCaml dependencies ───────────────────────────────────────
echo "Installing OCaml dependencies..."
eval "$(opam env 2>/dev/null)" || true
opam install --deps-only ./fstar.opam --yes

# ── Build F* + Pulse (stage 3) ───────────────────────────────────────
echo "Building F* + Pulse (make 3, -j$JOBS)..."
make -j"$JOBS" 3

# ── Build KaRaMeL ────────────────────────────────────────────────────
echo "Building KaRaMeL..."
make karamel

# ── Verify ───────────────────────────────────────────────────────────
echo ""
echo "═══ Build complete ═══"
echo "  fstar.exe : $TARGET_DIR/bin/fstar.exe"
echo "  krml      : $TARGET_DIR/karamel/krml"
"$TARGET_DIR/bin/fstar.exe" --version
