---
name: sourcebuild
description: Build F*, Pulse, and KaRaMeL from source (fstar2 branch) for use in a verification project
---

## Invocation

This skill is used when:
- Setting up a fresh F*/Pulse/KaRaMeL toolchain from source
- Writing a setup script for a project that depends on F*
- Understanding the fstar2 build system and directory layout
- Troubleshooting build failures

## Prerequisites

- **git**
- **opam** (OCaml package manager) with OCaml >= 4.14
- **Z3** (SMT solver) — multiple versions needed: 4.8.5 and 4.13.3 at minimum

### Installing opam and OCaml

```bash
# Install opam (if not present)
bash -c "sh <(curl -fsSL https://opam.ocaml.org/install.sh)"

# Initialize with a suitable OCaml compiler
opam init --compiler=ocaml.5.3.0 --disable-sandboxing
eval $(opam env)
```

### Installing Z3

F* ships a helper script to download the correct Z3 versions:

```bash
# After cloning (see below), install Z3 to a directory on your PATH
bash FStar/.scripts/get_fstar_z3.sh /usr/local/bin
```

This downloads Z3 4.8.5, 4.13.3, and 4.15.3 as `z3-4.8.5`, `z3-4.13.3`, `z3-4.15.3`.
F* selects the appropriate version automatically.

## The fstar2 Branch

The `fstar2` branch of `FStarLang/FStar` is a unified repository containing:

- **F\*** — the core language and compiler
- **Pulse** — separation-logic language extension (subdirectory `pulse/`)
- **KaRaMeL** — C extraction backend (git submodule `karamel/`)

This replaces the older setup where F*, Pulse, and KaRaMeL were separate repositories.

## Clone and Build

### Step 1: Clone

```bash
git clone --branch fstar2 git@github.com:FStarLang/FStar.git
cd FStar
git submodule update --init karamel
```

### Step 2: Install OCaml dependencies

```bash
opam install --deps-only ./fstar.opam --yes
```

This installs: batteries, zarith, stdint, yojson, dune, menhir, menhirLib,
memtrace, mtime, pprint, sedlex, ppxlib, process, ppx_deriving, ppx_deriving_yojson.

### Step 3: Build F* + Pulse

```bash
make -j$(nproc) 3
```

This runs the full bootstrap pipeline:
1. **Stage 0**: Pre-built F* compiler (checked into repo)
2. **Stage 1**: F* compiled by stage 0
3. **Stage 2**: F* compiled by stage 1 (the verified compiler)
4. **Stage 3**: Stage 2 + Pulse plugin compiled in + Pulse library verified

After `make 3`, the installed toolchain is at `stage3/out/` and `out/` is a
symlink to it:

```
FStar/
├── out -> stage3/out           # Active installation
│   ├── bin/fstar.exe           # F* compiler with Pulse plugin
│   └── lib/fstar/
│       ├── ulib/               # F* standard library sources
│       ├── ulib.checked/       # Pre-verified ulib .checked files
│       ├── pulse/              # Pulse library (common + pulse)
│       │   ├── common/         # PulseCore sources
│       │   ├── common.checked/
│       │   ├── pulse/          # Pulse.Lib sources
│       │   └── pulse.checked/
│       └── compiler/           # Compiled OCaml modules (.cmi/.cmx)
├── bin/fstar.exe -> out/bin/fstar.exe
├── karamel/                    # KaRaMeL submodule
│   └── krml                    # KaRaMeL binary (after step 4)
└── pulse/                      # Pulse source tree
```

### Step 4: Build KaRaMeL

```bash
make karamel
```

This runs `make minimal` in the karamel submodule, building the `krml` binary via dune.
It does **not** build krmllib (the KaRaMeL standard library `.krml` files) — see the
krmlextraction skill for how to handle stdlib types.

### Shortcut: Stage 2 Only

If you do not need Pulse, `make 2` builds just F* (faster):

```bash
make -j$(nproc) 2
```

This gives you `stage2/out/bin/fstar.exe` and `out -> stage2/out`.

## Using the Built Toolchain in a Project

After building, the key entry points are:
- `FStar/bin/fstar.exe` — F* compiler with Pulse plugin (symlink to `out/bin/fstar.exe`)
- `FStar/karamel/krml` — KaRaMeL C extraction tool

The stage3 `fstar.exe` finds Pulse library modules automatically — no extra `--include`
flags are needed for `Pulse.Lib.*`, `Pulse.Class.*`, etc.

See the `projectsetup` skill for directory layout, Makefile template, and verification flags.

## Setup Script Template

```bash
#!/usr/bin/env bash
set -euo pipefail

FSTAR_REPO="git@github.com:FStarLang/FStar.git"
FSTAR_BRANCH="fstar2"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
FSTAR_HOME="$SCRIPT_DIR/tools/FStar"
JOBS="${JOBS:-$(nproc 2>/dev/null || echo 4)}"

# Clone or update
if [ -d "$FSTAR_HOME/.git" ]; then
    git -C "$FSTAR_HOME" fetch origin "$FSTAR_BRANCH"
    git -C "$FSTAR_HOME" checkout "$FSTAR_BRANCH"
    git -C "$FSTAR_HOME" pull --ff-only
else
    mkdir -p "$(dirname "$FSTAR_HOME")"
    git clone --branch "$FSTAR_BRANCH" "$FSTAR_REPO" "$FSTAR_HOME"
fi

cd "$FSTAR_HOME"
git submodule update --init karamel

# Install OCaml deps
opam install --deps-only ./fstar.opam --yes

# Build F* + Pulse + KaRaMeL
make -j"$JOBS" 3
make karamel

echo "F* ready: $FSTAR_HOME/out/bin/fstar.exe"
"$FSTAR_HOME/out/bin/fstar.exe" --version
```

## Build Targets Reference

| Target | What it builds | Output |
|--------|---------------|--------|
| `make 0` | Stage 0 (pre-built, fast) | `stage0/out/bin/fstar.exe` |
| `make 1` | Stage 1 (F* built by stage 0) | `stage1/out/bin/fstar.exe` |
| `make 2` | Stage 2 (F* built by stage 1) | `stage2/out/bin/fstar.exe` |
| `make 3` | Stage 3 (stage 2 + Pulse) | `stage3/out/bin/fstar.exe` |
| `make build` | Alias for `make 3` | |
| `make karamel` | KaRaMeL binary only | `karamel/krml` |
| `make all` | Full build + tests | |
| `make setlink-N` | Point `out/` → `stageN/out/` | `out` symlink |

## Troubleshooting

### opam dependency conflict

**Symptom**: `opam install --deps-only ./fstar.opam` fails with version conflicts.

**Fix**: Create a fresh opam switch:
```bash
opam switch create fstar2 ocaml.5.3.0
eval $(opam env)
opam install --deps-only ./fstar.opam --yes
```

### Z3 not found or wrong version

**Symptom**: `Could not find a suitable Z3` or `Z3 version mismatch`.

**Fix**: Install the required Z3 versions and ensure they are on PATH as `z3-4.8.5`,
`z3-4.13.3`, etc.:
```bash
bash FStar/.scripts/get_fstar_z3.sh "$HOME/.local/bin"
export PATH="$HOME/.local/bin:$PATH"
```

### Stage 0 stale or corrupt

**Symptom**: Stage 1 build fails with strange OCaml errors.

**Fix**: Clean stage 0 and rebuild:
```bash
make -C stage0 clean
make 0
make -j$(nproc) 3
```

### Pulse build uses wrong fstar.exe

**Symptom**: Pulse build fails with syntax errors or rejected code.

**Cause**: An older F* from opam or a previous stage is on PATH and gets picked up
instead of the just-built stage2/stage3.

**Fix**: `make 3` handles this correctly by passing `FSTAR_EXE` explicitly. If building
Pulse manually, always specify:
```bash
make -C pulse local-install FSTAR_EXE=$(pwd)/out/bin/fstar.exe
```

### KaRaMeL krmllib build fails (--cmi flag)

**Symptom**: `make krmllib` fails with `Unknown option: --cmi`.

**Cause**: The `--cmi` flag was removed from F* (cross-module inlining is now always on).
The karamel submodule's `krmllib/Makefile` may still reference it.

**Fix**: Remove `--cmi` from `karamel/krmllib/Makefile`. Note: `make karamel` (which
builds only the `krml` binary) is unaffected — this only matters if you need `krmllib/.extract/`.

### make clean in Pulse deletes tracked files

**Symptom**: After `make -C pulse clean`, dune project files under `pulse/build/ocaml/`
are missing.

**Fix**: Restore them from git:
```bash
git checkout pulse/build/ocaml/
```

## Additional Resources

- [F* GitHub repository](https://github.com/FStarLang/FStar)
- [F* tutorial](https://fstar-lang.org/tutorial/)
- [Pulse documentation](https://github.com/FStarLang/FStar/tree/fstar2/pulse)
- See the `krmlextraction` skill for extracting verified code to C
- See the `fstarmcp` skill for incremental typechecking via the MCP server

## Building the F* MCP Server

The F* MCP server enables incremental typechecking (see the `fstarmcp` skill).
It is a Rust project that requires `cargo` (the Rust build tool).

### Prerequisites

Install Rust via [rustup](https://rustup.rs/) if not already installed:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"
```

### Clone and Build

```bash
# Clone alongside the FStar repo
git clone https://github.com/FStarLang/fstar-mcp.git
cd fstar-mcp

# Build (release mode)
cargo build --release
```

The binary is at `fstar-mcp/target/release/fstar-mcp`.

### Add to Setup Script

Extend the setup script template above with:

```bash
# --- F* MCP Server ---
FSTARMCP_REPO="https://github.com/FStarLang/fstar-mcp.git"
FSTARMCP_HOME="$SCRIPT_DIR/tools/fstar-mcp"

if [ -d "$FSTARMCP_HOME/.git" ]; then
    git -C "$FSTARMCP_HOME" pull --ff-only
else
    git clone "$FSTARMCP_REPO" "$FSTARMCP_HOME"
fi

cd "$FSTARMCP_HOME"
cargo build --release
echo "fstar-mcp ready: $FSTARMCP_HOME/target/release/fstar-mcp"
```
