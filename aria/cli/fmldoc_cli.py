"""CLI for logic constraint formats: translate, validate, and analyze.

Supported formats (for validate/analyze): dimacs, smtlib2.
Supported translations: dimacs -> smtlib2.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

# Translation pairs (input_format, output_format) that are implemented.
SUPPORTED_TRANSLATIONS: List[Tuple[str, str]] = [
    ("dimacs", "smtlib2"),
]


def detect_format(filename: str) -> str:
    """Auto-detect format from file extension"""
    ext_map = {
        ".cnf": "dimacs",
        ".qdimacs": "qdimacs",
        ".tplp": "tplp",
        ".fzn": "fzn",
        ".smt2": "smtlib2",
        ".sy": "sygus",
        ".lp": "lp",
        ".dl": "datalog",
    }

    ext = Path(filename).suffix.lower()
    return ext_map.get(ext)


def handle_translate(args) -> int:
    """Handle translation between formats."""
    input_format = args.input_format
    output_format = args.output_format

    if args.auto_detect:
        if not input_format:
            input_format = detect_format(args.input_file)
        if not output_format:
            output_format = detect_format(args.output_file)

    if not input_format or not output_format:
        raise ValueError("Input and output formats must be specified or auto-detected")

    pair = (input_format.lower(), output_format.lower())
    if pair == ("dimacs", "smtlib2"):
        from aria.translator.dimacs2smt import (
            convert_dimacs_to_smt2,
        )  # pylint: disable=import-outside-toplevel

        convert_dimacs_to_smt2(args.input_file, args.output_file)
        return 0

    supported = ", ".join(f"{i} -> {o}" for i, o in SUPPORTED_TRANSLATIONS)
    raise ValueError(
        f"No translator for {input_format} -> {output_format}. "
        f"Supported: {supported}"
    )


def handle_validate(args) -> int:
    """Validate file format."""
    with open(args.input_file, encoding="utf-8") as f:
        content = f.read()

    fmt = args.format or detect_format(args.input_file)
    if not fmt:
        raise ValueError(
            "Could not detect format from file extension. Use -f/--format."
        )

    try:
        if fmt == "smtlib2":
            import z3  # pylint: disable=import-outside-toplevel

            z3.parse_smt2_string(content)
        elif fmt == "dimacs":
            lines = [
                line
                for line in content.splitlines()
                if line.strip() and not line.strip().startswith("c")
            ]
            p_lines = [line for line in lines if line.strip().startswith("p ")]
            if not p_lines:
                raise ValueError("Missing problem line (p cnf ...)")
            parts = p_lines[0].split()
            if len(parts) < 4 or parts[0] != "p" or parts[1] != "cnf":
                raise ValueError("Invalid problem line: expected 'p cnf <vars> <clauses>'")
            int(parts[2])
            int(parts[3])
        else:
            raise ValueError(f"Validation not implemented for format: {fmt}")

        print(f"Successfully validated {args.input_file}")
        return 0
    except (ValueError, IOError, OSError) as e:
        print(f"Validation failed: {e}", file=sys.stderr)
        return 1


def handle_analyze(args) -> int:
    """Analyze constraint properties."""
    fmt = args.format or detect_format(args.input_file)
    if not fmt:
        raise ValueError(
            "Could not detect format from file extension. Use -f/--format."
        )

    with open(args.input_file, encoding="utf-8") as f:
        content = f.read()

    if fmt == "dimacs":
        lines = [
            line.strip()
            for line in content.splitlines()
            if line.strip() and not line.strip().startswith("c")
        ]
        p_lines = [line for line in lines if line.startswith("p cnf")]
        if not p_lines:
            print("Error: Missing problem line (p cnf ...)", file=sys.stderr)
            return 1
        parts = p_lines[0].split()
        if len(parts) < 4:
            print("Error: Invalid problem line", file=sys.stderr)
            return 1
        num_vars = int(parts[2])
        num_clauses = int(parts[3])
        print(f"Number of variables: {num_vars}")
        print(f"Number of clauses: {num_clauses}")

    elif fmt == "smtlib2":
        decls = sum(
            1
            for line in content.splitlines()
            if line.strip().startswith("(declare-")
        )
        asserts = sum(
            1
            for line in content.splitlines()
            if line.strip().startswith("(assert")
        )
        print(f"Number of declarations: {decls}")
        print(f"Number of assertions: {asserts}")

    else:
        print(f"Analysis not implemented for format: {fmt}", file=sys.stderr)
        return 1

    return 0


def handle_formats(_args: argparse.Namespace) -> int:
    """Print supported formats and translation pairs."""
    print("Supported validation/analysis formats: dimacs, smtlib2")
    print("Supported translations:")
    for i, o in SUPPORTED_TRANSLATIONS:
        print(f"  {i} -> {o}")
    return 0


def handle_batch(args) -> int:
    """Handle batch processing."""
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process all files
    success = 0
    failed = 0

    for input_file in input_dir.glob("*"):
        if input_file.is_file():
            try:
                # Auto-detect formats if needed
                in_format = args.input_format or detect_format(str(input_file))
                if not in_format:
                    continue

                out_format = args.output_format or args.input_format
                if not out_format:
                    continue

                # Construct output path
                ext_map = {
                    "dimacs": ".cnf",
                    "qdimacs": ".qdimacs",
                    "tplp": ".tplp",
                    "fzn": ".fzn",
                    "smtlib2": ".smt2",
                    "sygus": ".sy",
                    "lp": ".lp",
                    "datalog": ".dl",
                }
                out_ext = ext_map.get(out_format, ".txt")
                output_file = output_dir / (input_file.stem + out_ext)

                # Translate
                translate_args = argparse.Namespace(
                    input_format=in_format,
                    output_format=out_format,
                    input_file=str(input_file),
                    output_file=str(output_file),
                    auto_detect=False,
                    preserve_comments=(
                        args.preserve_comments
                        if hasattr(args, "preserve_comments")
                        else False
                    ),
                )

                if handle_translate(translate_args) == 0:
                    success += 1
                else:
                    failed += 1

            except (ValueError, IOError, OSError) as e:
                print(f"Error processing {input_file}: {e}")
                failed += 1

    print(f"Batch processing complete: {success} succeeded, {failed} failed")
    return 0 if failed == 0 else 1


def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Logic constraint format translator, validator, and analyzer."
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("-d", "--debug", action="store_true", help="Debug output")

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Formats (list supported)
    subparsers.add_parser("formats", help="List supported formats and translations")

    # Translate
    t = subparsers.add_parser("translate", help="Translate between formats")
    t.add_argument("-i", "--input-file", required=True, help="Input file")
    t.add_argument("-o", "--output-file", required=True, help="Output file")
    t.add_argument("--input-format", help="Input format")
    t.add_argument("--output-format", help="Output format")
    t.add_argument("--auto-detect", action="store_true", help="Auto-detect formats")

    # Validate
    v = subparsers.add_parser("validate", help="Validate file format")
    v.add_argument("-i", "--input-file", required=True, help="Input file")
    v.add_argument("-f", "--format", help="File format")

    # Analyze
    a = subparsers.add_parser("analyze", help="Analyze properties")
    a.add_argument("-i", "--input-file", required=True, help="Input file")
    a.add_argument("-f", "--format", help="File format")

    # Batch
    b = subparsers.add_parser("batch", help="Batch process")
    b.add_argument("-i", "--input-dir", required=True, help="Input directory")
    b.add_argument("-o", "--output-dir", required=True, help="Output directory")
    b.add_argument("--input-format", help="Input format")
    b.add_argument("--output-format", help="Output format")

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Main entry point for the CLI."""
    parser = create_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 1

    command_handlers = {
        "translate": handle_translate,
        "validate": handle_validate,
        "analyze": handle_analyze,
        "batch": handle_batch,
        "formats": handle_formats,
    }

    try:
        handler = command_handlers.get(args.command)
        if handler:
            return handler(args)
        return 0
    except (ValueError, IOError, OSError) as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.debug:
            raise
        return 1


if __name__ == "__main__":
    sys.exit(main())
