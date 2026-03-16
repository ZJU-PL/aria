"""
CHC(BV) to SyGuS(BV) converter
"""

import sys
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from format_converter import UnifiedConverter, ConversionType
from conversion_utils import FileProcessor


def main():
    parser = argparse.ArgumentParser(description="Convert CHC BV to SyGuS BV")
    parser.add_argument("input_path", help="Input file or directory")
    parser.add_argument("output_dir", help="Output directory")
    parser.add_argument(
        "--width", type=int, default=32, help="Bitvector width (default: 32)"
    )
    parser.add_argument(
        "--signedness",
        choices=["signed", "unsigned"],
        default="signed",
        help="Bitvector signedness (default: signed)",
    )

    args = parser.parse_args()

    # Use the unified converter
    converter = UnifiedConverter(
        ConversionType.CHC_BV_TO_SYGUS_BV, args.width, args.signedness
    )

    files = FileProcessor.get_files_with_extensions(args.input_path, [".smt2", ".sl"])

    for filename in files:
        result = FileProcessor.safe_process_file(filename, converter.convert)
        if result:
            import os

            filename_base = os.path.basename(filename)
            output_path = os.path.join(
                args.output_dir,
                f"{filename_base}_{args.width}bits_{args.signedness}.sl",
            )
            FileProcessor.write_output_file(result, output_path)


if __name__ == "__main__":
    main()
