---
name: fstarmcp
description: Use the F* MCP server for interactive, incremental typechecking of F* and Pulse code
---

## Overview

The F* MCP server (`fstar-mcp`) provides an HTTP API wrapping F*'s `--ide` protocol.
It enables **incremental typechecking**: create a session once, then re-typecheck modified
code without restarting F* or reloading dependencies. This dramatically speeds up iterative
proof development.

Source: [FStarLang/fstar-mcp](https://github.com/FStarLang/fstar-mcp)

See the `sourcebuild` skill for instructions on building `fstar-mcp` from source.

## MCP Registration

Register the server in `.copilot/mcp-config.json` at your project root:

```json
{
  "mcpServers": {
    "fstar-mcp": {
      "type": "http",
      "url": "http://localhost:3001/"
    }
  }
}
```

Copilot CLI auto-connects when the server is running.

## Starting the Server

```bash
# Start on port 3001 (matches mcp-config.json)
FSTAR_MCP_PORT=3001 path/to/fstar-mcp &

# With debug logging
RUST_LOG=fstar_mcp=debug FSTAR_MCP_PORT=3001 path/to/fstar-mcp &
```

The server runs on `http://127.0.0.1:3001`. All API calls are JSON-RPC 2.0 POST
requests to `/`.

## API Protocol

All requests use JSON-RPC 2.0 over HTTP POST:

```bash
curl -s -X POST http://localhost:3001/ \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json' \
  -d '{"jsonrpc": "2.0", "method": "tools/call", "id": 1, "params": {"name": "TOOL_NAME", "arguments": {...}}}'
```

Responses contain `result.content[0].text` with a JSON string that must be parsed again.

## Workflow

### 1. Create a Session

```bash
curl -s -X POST http://localhost:3001/ \
  -H 'Content-Type: application/json' \
  -d '{"jsonrpc":"2.0","method":"tools/call","id":1,"params":{"name":"create_session","arguments":{
    "file_path": "/path/to/Module.fst",
    "cwd": "/project/root",
    "include_dirs": ["/path/to/lib", "/path/to/specs"],
    "options": ["--cache_dir", "/path/to/obj", "--already_cached", "Prims FStar"]
  }}}'
```

**Parameters:**
- `file_path` (string, optional): Path to the F* file. Must exist on disk. If omitted, creates a temp file.
- `fstar_exe` (string, optional): Path to `fstar.exe`. Defaults to `fstar.exe` in PATH.
- `cwd` (string, optional): Working directory. Defaults to file's directory.
- `include_dirs` (string[], optional): Directories for `--include`.
- `options` (string[], optional): Extra F* CLI options.

**Returns:** `session_id`, `status` ("ok"/"error"), `diagnostics[]`, `fragments[]`

The initial `create_session` typechecks the file's current on-disk contents. The returned
`session_id` is used for all subsequent operations.

### 2. Typecheck Modified Code

```bash
curl -s -X POST http://localhost:3001/ \
  -d '{"jsonrpc":"2.0","method":"tools/call","id":2,"params":{"name":"typecheck_buffer","arguments":{
    "session_id": "UUID",
    "code": "module Foo\nlet x = 42",
    "kind": "full"
  }}}'
```

**Parameters:**
- `session_id` (string, required): From `create_session`.
- `code` (string, required): Full module source code. Must start with matching `module` declaration.
- `lax` (boolean, optional): Shortcut for `kind: "lax"`.
- `kind` (string, optional): `"full"` (default), `"lax"`, `"cache"`, `"reload-deps"`, `"verify-to-position"`, `"lax-to-position"`.
- `to_line` / `to_column` (integer, optional): For position-based kinds.

**Returns:** `status`, `diagnostics[]`, `fragments[]`

Each fragment has `start_line`, `end_line`, `start_column`, `end_column`, and `status` ("ok"/"failed").

### 3. Look Up Symbols

```bash
curl -s -X POST http://localhost:3001/ \
  -d '{"jsonrpc":"2.0","method":"tools/call","id":3,"params":{"name":"lookup_symbol","arguments":{
    "session_id": "UUID",
    "file_path": "/path/to/Module.fst",
    "line": 10, "column": 5,
    "symbol": "my_function"
  }}}'
```

**Returns:** `kind` ("symbol"/"module"/"not_found"), `name`, `type_info`, `defined_at`.

### 4. Other Tools

| Tool | Description |
|------|-------------|
| `list_sessions` | List all active sessions with their file paths |
| `restart_solver` | Restart Z3 for a session (useful when solver gets stuck) |
| `get_proof_context` | Get proof obligations from tactic-based proofs |
| `update_buffer` | Add/update files in F*'s virtual file system (vfs-add) for dependency resolution |
| `close_session` | Clean up a session |

## Project Configuration

To use `fstar-mcp` with your project, determine the correct `include_dirs` and
`options` from your project's Makefile or build system. Common patterns:

### Standard F*/Pulse project (using fstar2 layout)

```json
{
  "file_path": "/project/src/Module.fst",
  "cwd": "/project",
  "include_dirs": ["src/spec", "src/impl", "obj"],
  "options": [
    "--cache_dir", "obj",
    "--already_cached", "Prims FStar Pulse.Lib PulseCore"
  ]
}
```

### Project with multiple source directories

```json
{
  "file_path": "/project/code/algo/Module.fst",
  "cwd": "/project",
  "include_dirs": [
    "lib", "specs",
    "code/algo", "code/utils",
    "obj",
    "../FStar/pulse/out/lib/pulse/lib",
    "../karamel/krmllib"
  ],
  "options": [
    "--cache_dir", "obj",
    "--already_cached", "Prims FStar PulseCore Pulse.Lib Pulse.Class Pulse.Main",
    "--ext", "pulse:rvalues",
    "--ext", "fly_deps",
    "--ext", "optimize_let_vc"
  ]
}
```

### Deriving options from a Makefile

If your project has a Makefile that drives F* verification, extract the flags:

```bash
# See what flags make passes to fstar.exe
make VERBOSE=1 verify-module 2>&1 | grep fstar.exe

# Common Makefile variables to look for:
# FSTAR_INCLUDES = --include src/spec --include src/impl
# FSTAR_OPTIONS  = --cache_dir obj --already_cached "Prims FStar"
```

Map `--include` flags to `include_dirs` and the rest to `options`.

## Tips

- **Incremental workflow**: Create session once (slow, loads deps), then `typecheck_buffer`
  repeatedly (fast, only re-checks changed fragments).
- **Module name must match file path**: The `module` declaration in `code` must match the
  file name from `create_session` (e.g., file `Spec.Poly1305.fst` → `module Spec.Poly1305`).
- **Pulse files**: No special configuration beyond including Pulse library dirs. The `#lang-pulse`
  directive in the file is handled automatically.
- **Lax mode**: Use `"lax": true` for fast syntax/type checking without SMT verification.
  Useful for validating structure before committing to full verification.
- **Position-based verification**: Use `"kind": "verify-to-position"` with `to_line` to verify
  only up to a specific point — useful for large files where you're working on one function.
- **Symbol lookup**: Works after a successful typecheck. Useful for finding types and
  definition locations of library functions you need to call.
- **Session replacement**: Creating a new session with the same `file_path` replaces the old one.
- **Restart solver**: When Z3 gets stuck or accumulates bad state, use `restart_solver` rather
  than recreating the entire session (which would reload all dependencies).
