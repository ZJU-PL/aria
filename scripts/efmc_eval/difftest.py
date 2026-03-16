import os
import glob
import time
import argparse
import logging
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from efmc.sts import TransitionSystem
from efmc.utils.verification_utils import VerificationResult
from efmc.tests.sts_generator import TransitionSystemGenerator
from efmc.frontends import parse_chc, parse_sygus

from efmc.engines.pdr.pdr_prover import PDRProver
from efmc.engines.kinduction.kind_prover import KInductionProver
from efmc.engines.bdd.bdd_prover import BDDProver
from efmc.engines.ef.ef_prover import EFProver
from efmc.engines.qi.qi_prover import QuantifierInstantiationProver
from efmc.engines.houdini.houdini_prover import HoudiniProver
from efmc.engines.abduction.abduction_prover import AbductionProver
from efmc.engines.predabs.predabs_prover import PredicateAbstractionProver
from efmc.engines.symabs.symabs_prover import SymbolicAbstractionProver

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("difftest")


@dataclass
class EngineConfig:
    name: str
    engine_class: type
    supported_types: List[str]
    k_value: int = None
    timeout: int = None
    additional_params: Dict[str, Any] = None


@dataclass
class TestResult:
    test_name: str
    engine_name: str
    result: VerificationResult
    execution_time: float
    error: Optional[str] = None


class DiffTester:
    def __init__(self, config_file: Optional[str] = None):
        self.engines = self._get_default_engines()
        self.test_results: List[TestResult] = []
        self.generator = TransitionSystemGenerator()
        self.systems_map: Dict[str, TransitionSystem] = {}

        if config_file:
            self._load_config(config_file)

    def _get_default_engines(self) -> Dict[str, EngineConfig]:
        return {
            "pdr": EngineConfig(
                "PDR", PDRProver, ["bool", "int", "real", "bv"], timeout=60
            ),
            "kind": EngineConfig(
                "K-Induction",
                KInductionProver,
                ["bool", "int", "real", "bv"],
                k_value=30,
            ),
            "bdd": EngineConfig("BDD", BDDProver, ["bool"]),
            "ef": EngineConfig(
                "EF",
                EFProver,
                ["bool", "int", "real", "bv"],
                additional_params={"template_type": "interval"},
            ),
            "qi": EngineConfig(
                "QI", QuantifierInstantiationProver, ["bool", "int", "real", "bv"]
            ),
            "houdini": EngineConfig(
                "Houdini", HoudiniProver, ["bool", "int", "real", "bv"]
            ),
            "abduction": EngineConfig(
                "Abduction", AbductionProver, ["bool", "int", "real"]
            ),
            "predabs": EngineConfig(
                "Predicate Abstraction",
                PredicateAbstractionProver,
                ["bool", "int", "real", "bv"],
            ),
            "symabs": EngineConfig(
                "Symbolic Abstraction", SymbolicAbstractionProver, ["bool", "bv"]
            ),
        }

    def _load_config(self, config_file: str):
        try:
            with open(config_file, "r") as f:
                config = json.load(f)
            for engine_name, engine_config in config.get("engines", {}).items():
                if engine_name in self.engines:
                    for key, value in engine_config.items():
                        setattr(self.engines[engine_name], key, value)
                        if (
                            key == "additional_params"
                            and self.engines[engine_name].additional_params
                        ):
                            self.engines[engine_name].additional_params.update(value)
            logger.info(f"Loaded config from {config_file}")
        except Exception as e:
            logger.error(f"Failed to load config: {e}")

    def _get_system_type(self, system: TransitionSystem) -> str:
        for attr, type_name in [
            ("has_bool", "bool"),
            ("has_int", "int"),
            ("has_real", "real"),
            ("has_bv", "bv"),
        ]:
            if getattr(system, attr):
                return type_name
        return "unknown"

    def _run_engine(
        self, engine_config: EngineConfig, system: TransitionSystem
    ) -> Tuple[VerificationResult, float, Optional[str]]:
        start_time = time.time()
        try:
            engine = engine_config.engine_class(
                system, **(engine_config.additional_params or {})
            )
            if engine_config.name == "K-Induction":
                result = engine.solve(k=engine_config.k_value or 30)
            elif engine_config.name == "PDR" and engine_config.timeout:
                result = engine.solve(timeout=engine_config.timeout)
            else:
                result = engine.solve()
        except Exception as e:
            return None, time.time() - start_time, str(e)

        execution_time = time.time() - start_time
        if engine_config.timeout and execution_time > engine_config.timeout:
            result = (
                VerificationResult(False, None, None, True) if not result else result
            )
            result.is_unknown = True
            return result, execution_time, f"Timeout after {execution_time:.2f}s"

        return result, execution_time, None

    def load_systems_from_directory(
        self, directory: str, file_type: str = "smt2"
    ) -> List[Tuple[str, TransitionSystem]]:
        systems = []
        for file_path in glob.glob(os.path.join(directory, f"*.{file_type}")):
            try:
                file_name = os.path.basename(file_path)
                parser = parse_chc if file_type == "smt2" else parse_sygus
                all_vars, init, trans, post = parser(file_path, to_real_type=False)

                system = TransitionSystem()
                system.from_z3_cnts([all_vars, init, trans, post])

                if system and system.initialized:
                    systems.append((file_name, system))
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")

        logger.info(f"Loaded {len(systems)} systems from {directory}")
        return systems

    def generate_random_systems(
        self, count: int, types: List[str] = None
    ) -> List[Tuple[str, TransitionSystem]]:
        systems = []
        types = types or ["bool", "int", "real", "bv"]
        generators = {
            "bool": self.generator.gen_bool_program,
            "int": self.generator.gen_lia_program,
            "real": self.generator.gen_lra_program,
            "bv": self.generator.gen_bv_program,
        }

        for i in range(count):
            try:
                system_type = types[i % len(types)]
                system = generators[system_type]()
                systems.append((f"random_{system_type}_{i}", system))
            except Exception as e:
                logger.error(f"Error generating random system: {e}")

        logger.info(f"Generated {len(systems)} random systems")
        return systems

    def run_tests(
        self, systems: List[Tuple[str, TransitionSystem]], engines: List[str] = None
    ):
        engines = engines or list(self.engines.keys())
        for system_name, system in systems:
            self.systems_map[system_name] = system
            system_type = self._get_system_type(system)

            for engine_name in engines:
                if (
                    engine_name not in self.engines
                    or system_type not in self.engines[engine_name].supported_types
                ):
                    continue

                engine_config = self.engines[engine_name]
                result, execution_time, error = self._run_engine(engine_config, system)

                self.test_results.append(
                    TestResult(
                        system_name, engine_config.name, result, execution_time, error
                    )
                )

                status = (
                    "unknown"
                    if not result or result.is_unknown
                    else ("safe" if result.is_safe else "unsafe")
                )
                logger.info(
                    f"{engine_config.name} on {system_name}: {status} ({execution_time:.2f}s)"
                )

    def analyze_results(self) -> List[Dict[str, Any]]:
        potential_bugs = []
        test_cases = {}

        for result in self.test_results:
            test_cases.setdefault(result.test_name, []).append(result)

        for test_name, results in test_cases.items():
            if len(results) < 2:
                continue

            safe_engines = []
            unsafe_engines = []
            unknown_engines = []

            for result in results:
                if result.error or not result.result:
                    continue
                if result.result.is_unknown:
                    unknown_engines.append(result.engine_name)
                elif result.result.is_safe:
                    safe_engines.append(result.engine_name)
                else:
                    unsafe_engines.append(result.engine_name)

            if safe_engines and unsafe_engines:
                potential_bugs.append(
                    {
                        "test_name": test_name,
                        "safe_engines": safe_engines,
                        "unsafe_engines": unsafe_engines,
                        "unknown_engines": unknown_engines,
                        "severity": "high",
                    }
                )
                logger.warning(
                    f"Bug in {test_name}: Safe={safe_engines}, Unsafe={unsafe_engines}"
                )

        return potential_bugs

    def save_problematic_systems(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        potential_bugs = self.analyze_results()

        for bug in potential_bugs:
            test_name = bug["test_name"]
            if test_name in self.systems_map:
                system = self.systems_map[test_name]
                with open(os.path.join(output_dir, f"{test_name}.smt2"), "w") as f:
                    f.write(system.to_smt2())
                with open(
                    os.path.join(output_dir, f"{test_name}_details.json"), "w"
                ) as f:
                    json.dump(bug, f, indent=2)

        logger.info(f"Saved {len(potential_bugs)} problematic systems to {output_dir}")

    def generate_report(self, output_file: str = "difftest_report.json"):
        potential_bugs = self.analyze_results()
        report = {
            "summary": {
                "total_tests": len(set(r.test_name for r in self.test_results)),
                "total_engines": len(set(r.engine_name for r in self.test_results)),
                "total_results": len(self.test_results),
                "potential_bugs": len(potential_bugs),
            },
            "results": [
                {
                    "test_name": r.test_name,
                    "engine_name": r.engine_name,
                    "status": (
                        "unknown"
                        if not r.result or r.result.is_unknown
                        else ("safe" if r.result.is_safe else "unsafe")
                    ),
                    "execution_time": r.execution_time,
                    "error": r.error,
                }
                for r in self.test_results
            ],
            "potential_bugs": potential_bugs,
        }

        with open(output_file, "w") as f:
            json.dump(report, f, indent=2)

        print(f"\nTest Summary:")
        print(
            f"Tests: {report['summary']['total_tests']}, Engines: {report['summary']['total_engines']}"
        )
        print(
            f"Results: {report['summary']['total_results']}, Bugs: {len(potential_bugs)}"
        )

        if potential_bugs:
            print("\nPotential Bugs:")
            for i, bug in enumerate(potential_bugs):
                print(
                    f"{i+1}. {bug['test_name']}: Safe={bug['safe_engines']}, Unsafe={bug['unsafe_engines']}"
                )


def main():
    parser = argparse.ArgumentParser(
        description="Differential testing for EFMC engines"
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--dir", help="Directory with test files")
    input_group.add_argument("--random", type=int, help="Number of random tests")

    parser.add_argument("--file-type", choices=["smt2", "sl"], default="smt2")
    parser.add_argument("--engines", nargs="+", help="Engines to test")
    parser.add_argument(
        "--types",
        nargs="+",
        choices=["bool", "int", "real", "bv"],
        default=["bool", "int", "real", "bv"],
    )
    parser.add_argument("--config", help="Config file path")
    parser.add_argument(
        "--k-value", type=int, default=30, help="K value for K-Induction"
    )
    parser.add_argument(
        "--output", default="difftest_report.json", help="Output report file"
    )
    parser.add_argument("--save-bugs", help="Directory to save problematic systems")
    parser.add_argument("--timeout", type=int, default=60, help="Timeout in seconds")

    args = parser.parse_args()

    tester = DiffTester(args.config)

    if "kind" in tester.engines:
        tester.engines["kind"].k_value = args.k_value
    if "pdr" in tester.engines:
        tester.engines["pdr"].timeout = args.timeout

    systems = (
        tester.load_systems_from_directory(args.dir, args.file_type)
        if args.dir
        else tester.generate_random_systems(args.random, args.types)
    )

    if not systems:
        logger.error("No test systems available")
        return

    tester.run_tests(systems, args.engines)
    tester.generate_report(args.output)

    if args.save_bugs:
        tester.save_problematic_systems(args.save_bugs)


if __name__ == "__main__":
    main()
