import argparse
import json
from pathlib import Path


def _parse_metrics_line(line: str) -> dict[str, str]:
    marker = "pcdclt_metrics "
    if marker not in line:
        raise ValueError("metric record missing")
    payload = line.split(marker, 1)[1].strip()
    parsed: dict[str, str] = {}
    for field in payload.split():
        key, value = field.split("=", 1)
        parsed[key] = value
    return parsed


def extract_metrics(log_path: Path, benchmark: str, exit_status: int) -> dict[str, object]:
    metric_line = None
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if "pcdclt_metrics " in line:
            metric_line = line
    if metric_line is None:
        raise ValueError(f"no pcdclt metric record found in {log_path}")

    parsed = _parse_metrics_line(metric_line)
    return {
        "benchmark": benchmark,
        "exit_status": exit_status,
        "generated_total": int(parsed["generated_total"]),
        "added_total": int(parsed["added_total"]),
        "theory_unsat_checks_total": int(parsed["theory_unsat_checks_total"]),
        "theory_sat_checks_total": int(parsed["theory_sat_checks_total"]),
        "theory_unknown_or_error_total": int(parsed["theory_unknown_or_error_total"]),
        "completed_rounds_total": int(parsed["completed_rounds_total"]),
        "wall_time_seconds": float(parsed["wall_time_seconds"]),
        "sampling_strategy": parsed["sampling_strategy"],
        "simplify_clauses": parsed["simplify_clauses"] == "True",
        "worker_count": int(parsed["worker_count"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--exit-status", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    data = extract_metrics(Path(args.log), args.benchmark, args.exit_status)
    output_path = Path(args.output)
    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
