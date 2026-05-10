# F* Proof Copilot

A [GitHub Copilot CLI](https://docs.github.com/copilot/concepts/agents/about-copilot-cli) plugin for [F*](https://fstar-lang.org) proof-oriented programming.

## What's Included

### Agents

| Agent | Description |
|-------|-------------|
| **fstar-coder** | An expert F* and Pulse programmer that authors formal specifications, implements solutions, and proves correctness — all verified with `fstar.exe`. Handles both pure F* and Pulse (concurrent separation logic) uniformly. |

### Skills

| Skill | Description |
|-------|-------------|
| **smtprofiling** | Debug F* queries sent to Z3, diagnosing proof instability and performance issues. Includes a catalog of 10 proven stabilization techniques mined from real verification projects. |
| **proofdebugging** | Systematic workflows for debugging F*/Pulse verification failures — isolating failures, factoring lemmas, and hardening proofs. |
| **fstarverifier** | Verify F* and Pulse code with `fstar.exe` and interpret common error patterns. |
| **specreview** | Review F*/Pulse specifications for completeness, strength, and usability — catch weak postconditions and missing spec-impl connections. |
| **fstarmcp** | Use the F* MCP server for interactive, incremental typechecking — create a session once, then re-typecheck modified code without restarting F*. |
| **projectsetup** | Structure a new F*/Pulse verification project with Makefile and directory layout. |
| **sourcebuild** | Build F*, Pulse, and KaRaMeL from source (fstar2 branch). |
| **krmlextraction** | Extract verified F*/Pulse code to C via KaRaMeL. |

## Prerequisites

- **GitHub Copilot CLI** — Install from [github.com/github/copilot-cli](https://github.com/github/copilot-cli).
- **F\*** — The agent can build F* from source using the `sourcebuild` skill (requires `git`, `opam`/OCaml ≥ 4.14, and `Z3`). Alternatively, install a pre-built release from [fstar-lang.org](https://fstar-lang.org). Either way, `fstar.exe` must be on your PATH or configured via `FStar.fst.config.json`.
- **Z3** — The SMT solver used by F*. The `sourcebuild` skill can install the required versions automatically, or install alongside F* from a release.
- **Rust/Cargo** (optional) — Required only for building the F* MCP server for incremental typechecking. See the `sourcebuild` and `fstarmcp` skills.

## Installation

```bash
copilot plugin install FStarLang/proof-copilot
```

## Usage

### Using the FStarCoder agent

You can invoke the agent in several ways:

1. **Via the `/agent` command** — type `/agent` in an interactive session and select `fstar-coder`.

2. **Naturally in a prompt** — mention the agent by name:
   ```
   Use the fstar-coder agent to implement a verified binary search
   ```

3. **Via command line**:
   ```bash
   copilot --agent=fstar-coder --prompt "Implement a verified binary search over a sorted sequence"
   ```

### Using skills

Skills are automatically invoked when relevant, or can be called directly:

```
Use the smtprofiling skill to diagnose why this proof is slow
```

```
Use the specreview skill to check if my postconditions are strong enough
```

```
Use the proofdebugging skill to isolate this verification failure
```

## Roadmap

Future versions will add:
- smart context retrieval with vector embeddings
- enhanced spec review with provable tests
- support for other F*-related tools, EverParse, Kuiper, etc.

## License

Apache 2.0 — see [LICENSE](LICENSE).
