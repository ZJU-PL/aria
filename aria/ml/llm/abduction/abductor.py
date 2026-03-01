"""Abduction loop: propose psi with LLM, verify with Z3."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Protocol, Tuple

import z3

from aria.utils.z3_solver_utils import is_entail, is_sat

from .compiler import NLAbductionCompiler
from .data_structures import (
    AbductionIteration,
    AbductionResult,
    CompiledAbductionProblem,
    HybridHypothesis,
)
from .json_extract import extract_json_object, JsonExtractError
from .prompts import (
    create_counterexample_exchange_prompt,
    create_hypothesis_feedback_prompt,
    create_hypothesis_prompt,
)
from .smt import SmtParseError, build_env, parse_bool_term


def _model_to_assignment(
    model: z3.ModelRef, problem: CompiledAbductionProblem
) -> Dict[str, Any]:
    env = build_env(problem.variables)
    out: Dict[str, Any] = {}
    for name, z3v in env.z3_vars.items():
        out[name] = model.eval(z3v, model_completion=True)
    return out


class NLAbductor:
    """End-to-end NL -> compiled SMT -> hypothesis psi with feedback."""

    def __init__(
        self,
        llm: "LLMClient",
        compiler: Optional[NLAbductionCompiler] = None,
        max_iterations: int = 6,
        max_exchange_rounds: int = 3,
    ) -> None:
        self.llm = llm
        self.compiler = compiler or NLAbductionCompiler(llm=llm, max_attempts=3)
        self.max_iterations = max_iterations
        self.max_exchange_rounds = max_exchange_rounds

    def abduce(self, text: str) -> AbductionResult:
        start = time.time()
        out = AbductionResult(execution_time=0.0)

        compilation = self.compiler.compile(text)
        out.compilation = compilation
        if compilation.problem is None:
            out.error = compilation.error or "Compilation failed"
            out.execution_time = time.time() - start
            return out

        problem = compilation.problem
        out.compiled = problem

        # Main loop
        prompt = create_hypothesis_prompt(problem)
        last_ce: Optional[Dict[str, Any]] = None

        for i in range(1, self.max_iterations + 1):
            it = AbductionIteration(iteration=i, prompt=prompt)
            out.iterations.append(it)

            llm_response, _, _ = self.llm.infer(prompt, True)
            it.llm_response = llm_response

            try:
                env = build_env(problem.variables)
                hypothesis = self._parse_hybrid_hypothesis(llm_response, env)
                it.hypothesis = hypothesis

                psi_smt = hypothesis.smt_conjunction()
                full = z3.And(problem.domain_constraints, problem.premise, psi_smt)
                it.is_consistent = is_sat(full)
                if not it.is_consistent:
                    it.is_sufficient = False
                    it.is_valid = False
                    last_ce = self._counterexample_inconsistent(problem, psi_smt)
                    it.counterexample = last_ce
                    prompt = create_hypothesis_feedback_prompt(problem, it, last_ce or {})
                    continue

                it.is_sufficient = is_entail(full, problem.conclusion)
                it.is_valid = it.is_consistent and it.is_sufficient
                if it.is_valid:
                    out.hypothesis = hypothesis
                    out.is_consistent = True
                    out.is_sufficient = True
                    out.is_valid = True
                    break

                last_ce = self._counterexample_insufficient(problem, psi_smt)
                it.counterexample = last_ce

                # Optional SMT/LLM exchange: if NL-only constraints reject the SMT counterexample,
                # ask LLM to propose SMT lemmas that rule it out and retry sufficiency.
                if last_ce is not None and hypothesis.nl_terms:
                    psi_smt_updated = self._exchange_and_strengthen(
                        problem=problem,
                        env=env,
                        it=it,
                        hypothesis=hypothesis,
                        counterexample=last_ce,
                    )
                    if psi_smt_updated is not None:
                        psi_smt = psi_smt_updated
                        full = z3.And(problem.domain_constraints, problem.premise, psi_smt)
                        it.is_consistent = is_sat(full)
                        if it.is_consistent:
                            it.is_sufficient = is_entail(full, problem.conclusion)
                            it.is_valid = it.is_consistent and it.is_sufficient
                            if it.is_valid:
                                out.hypothesis = hypothesis
                                out.is_consistent = True
                                out.is_sufficient = True
                                out.is_valid = True
                                break

                prompt = create_hypothesis_feedback_prompt(problem, it, last_ce or {})

            except (JsonExtractError, SmtParseError, z3.Z3Exception, ValueError) as e:
                it.error = str(e)
                # Force a structured repair attempt.
                last_ce = {"error": it.error}
                it.counterexample = last_ce
                prompt = (
                    prompt
                    + "\n\nYour previous output could not be parsed/checked. "
                    + "Return ONLY the required JSON object (psi_smt, psi_nl)."
                )

        # Finalize
        if out.hypothesis is not None and out.is_valid:
            pass
        else:
            last = out.iterations[-1] if out.iterations else None
            if last and last.error:
                out.error = last.error
            elif out.iterations:
                out.error = "Failed to find a valid hypothesis within iteration limit"
            else:
                out.error = "No iterations executed"

        out.execution_time = time.time() - start
        return out

    def _counterexample_inconsistent(
        self, problem: CompiledAbductionProblem, hypothesis_smt: z3.BoolRef
    ) -> Optional[Dict[str, Any]]:
        s = z3.Solver()
        s.add(problem.domain_constraints, problem.premise, z3.Not(hypothesis_smt))
        if s.check() != z3.sat:
            return None
        return _model_to_assignment(s.model(), problem)

    def _counterexample_insufficient(
        self, problem: CompiledAbductionProblem, hypothesis_smt: z3.BoolRef
    ) -> Optional[Dict[str, Any]]:
        s = z3.Solver()
        s.add(
            problem.domain_constraints,
            problem.premise,
            hypothesis_smt,
            z3.Not(problem.conclusion),
        )
        if s.check() != z3.sat:
            return None
        return _model_to_assignment(s.model(), problem)

    def _parse_hybrid_hypothesis(self, llm_response: str, env) -> HybridHypothesis:
        obj = extract_json_object(llm_response)
        psi_smt_raw = obj.get("psi_smt", [])
        psi_nl_raw = obj.get("psi_nl", [])
        if not isinstance(psi_smt_raw, list) or not isinstance(psi_nl_raw, list):
            raise ValueError("psi_smt and psi_nl must be lists")

        smt_terms: List[z3.BoolRef] = []
        for t in psi_smt_raw:
            if not isinstance(t, str):
                raise ValueError("psi_smt items must be strings")
            smt_terms.append(parse_bool_term(t, env))

        nl_terms: List[str] = []
        for t in psi_nl_raw:
            if not isinstance(t, str):
                raise ValueError("psi_nl items must be strings")
            tt = t.strip()
            if tt:
                nl_terms.append(tt)

        return HybridHypothesis(smt_terms=smt_terms, nl_terms=nl_terms)

    def _exchange_and_strengthen(
        self,
        problem: CompiledAbductionProblem,
        env,
        it: AbductionIteration,
        hypothesis: HybridHypothesis,
        counterexample: Dict[str, Any],
    ) -> Optional[z3.BoolRef]:
        psi_smt_terms: List[z3.BoolRef] = list(hypothesis.smt_terms)

        for _round in range(self.max_exchange_rounds):
            prompt = create_counterexample_exchange_prompt(
                problem=problem,
                psi_nl=hypothesis.nl_terms,
                counterexample=counterexample,
            )
            resp, _, _ = self.llm.infer(prompt, True)
            it.verifier_response = resp

            obj = extract_json_object(resp)
            verdict = obj.get("verdict", "")
            lemmas = obj.get("lemmas_smt", [])

            if verdict not in {"accept", "reject"}:
                raise ValueError("verdict must be 'accept' or 'reject'")
            if not isinstance(lemmas, list):
                raise ValueError("lemmas_smt must be a list")

            if verdict == "accept":
                return None

            lemma_terms: List[z3.BoolRef] = []
            lemma_strs: List[str] = []
            for l in lemmas:
                if not isinstance(l, str):
                    raise ValueError("lemmas_smt items must be strings")
                lemma_strs.append(l)
                lemma_terms.append(parse_bool_term(l, env))

            if not lemma_terms:
                return None

            it.bridge_lemmas_smt.extend(lemma_strs)
            psi_smt_terms.extend(lemma_terms)

            # If the counterexample is now ruled out, stop exchanging for this iteration.
            s = z3.Solver()
            s.add(
                problem.domain_constraints,
                problem.premise,
                z3.And(*psi_smt_terms) if psi_smt_terms else z3.BoolVal(True),
                z3.Not(problem.conclusion),
            )
            if s.check() != z3.sat:
                return z3.And(*psi_smt_terms) if psi_smt_terms else z3.BoolVal(True)

            # Otherwise, we found a new counterexample under strengthened SMT; update and continue.
            counterexample = _model_to_assignment(s.model(), problem)
            it.counterexample = counterexample

        return z3.And(*psi_smt_terms) if psi_smt_terms else z3.BoolVal(True)


class LLMClient(Protocol):
    def infer(self, message: str, is_measure_cost: bool = False) -> Tuple[str, int, int]:
        ...
