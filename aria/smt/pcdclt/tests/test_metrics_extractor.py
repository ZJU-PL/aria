import json
import subprocess
from pathlib import Path

from aria.tests import TestCase, main


class TestMetricsExtractor(TestCase):
    def setUp(self):
        self.fixture_dir = Path("/tmp/pcdclt_metrics_test")
        self.fixture_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.fixture_dir / "normal.log"
        self.output_path = self.fixture_dir / "metrics.json"

    def test_extracts_compact_json(self):
        self.log_path.write_text(
            "2026-03-15 INFO aria.smt.pcdclt.solver pcdclt_metrics generated_total=4 added_total=2 theory_unsat_checks_total=4 theory_sat_checks_total=1 theory_unknown_or_error_total=0 completed_rounds_total=2 wall_time_seconds=12.500 sampling_strategy=enum simplify_clauses=True worker_count=8\n",
            encoding="utf-8",
        )

        subprocess.run(
            [
                "/home/androidusr/workspace/project_paralle_smt/aria/venv/bin/python",
                "aria/scripts/extract_pcdclt_metrics.py",
                "--log",
                str(self.log_path),
                "--benchmark",
                "bench.smt2",
                "--exit-status",
                "124",
                "--output",
                str(self.output_path),
            ],
            check=True,
            cwd="/home/androidusr/workspace/project_paralle_smt",
        )

        data = json.loads(self.output_path.read_text(encoding="utf-8"))
        self.assertEqual(data["benchmark"], "bench.smt2")
        self.assertEqual(data["exit_status"], 124)
        self.assertEqual(data["generated_total"], 4)
        self.assertEqual(data["added_total"], 2)
        self.assertEqual(data["theory_unsat_checks_total"], 4)
        self.assertEqual(data["theory_sat_checks_total"], 1)
        self.assertEqual(data["theory_unknown_or_error_total"], 0)
        self.assertEqual(data["completed_rounds_total"], 2)
        self.assertEqual(data["wall_time_seconds"], 12.5)
        self.assertEqual(data["sampling_strategy"], "enum")
        self.assertTrue(data["simplify_clauses"])
        self.assertEqual(data["worker_count"], 8)

    def test_missing_metric_line_fails(self):
        self.log_path.write_text("no metrics here\n", encoding="utf-8")

        result = subprocess.run(
            [
                "/home/androidusr/workspace/project_paralle_smt/aria/venv/bin/python",
                "aria/scripts/extract_pcdclt_metrics.py",
                "--log",
                str(self.log_path),
                "--benchmark",
                "bench.smt2",
                "--exit-status",
                "0",
                "--output",
                str(self.output_path),
            ],
            check=False,
            cwd="/home/androidusr/workspace/project_paralle_smt",
            capture_output=True,
            text=True,
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("no pcdclt metric record found", result.stderr)


if __name__ == "__main__":
    main()
