#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable, List


TEMPLATE = """(set-logic UFNIA)
(declare-fun pow2 (Int) Int)
(declare-fun intand (Int Int Int) Int)
(declare-fun intor (Int Int Int) Int)
(declare-fun intxor (Int Int Int) Int)
(define-fun bitof ((k Int) (l Int) (a Int)) Int (mod (div a (pow2 l)) 2))
(define-fun int_all_but_msb ((k Int) (a Int)) Int (mod a (pow2 (- k 1))))
(define-fun intand_helper ((a Int) (b Int)) Int (ite (and (= a 1) (= b 1) ) 1 0))
(define-fun intor_helper ((a Int) (b Int)) Int (ite (and (= a 0) (= b 0) ) 0 1))
(define-fun intxor_helper ((a Int) (b Int)) Int (ite (or (and (= a 0) (=  b 1)) (and (= a 1) (= b 0))) 1 0))
(define-fun intmax ((k Int)) Int (- (pow2 k) 1))
(define-fun intmin ((k Int)) Int 0)
(define-fun in_range ((k Int) (x Int)) Bool (and (>= x 0) (<= x (intmax k))))
(define-fun intudivtotal ((k Int) (a Int) (b Int)) Int (ite (= b 0) (- (pow2 k) 1) (div a b) ))
(define-fun intmodtotal ((k Int) (a Int) (b Int)) Int (ite (= b 0) a (mod a b)))
(define-fun intneg ((k Int) (a Int)) Int (intmodtotal k (- (pow2 k) a) (pow2 k)))
(define-fun intnot ((k Int) (a Int)) Int (- (intmax k) a))
(define-fun intmins ((k Int)) Int (pow2 (- k 1)))
(define-fun intmaxs ((k Int)) Int (intnot k (intmins k)))
(define-fun intshl ((k Int) (a Int) (b Int)) Int (intmodtotal k (* a (pow2 b)) (pow2 k)))
(define-fun intlshr ((k Int) (a Int) (b Int)) Int (intmodtotal k (intudivtotal k a (pow2 b)) (pow2 k)))
(define-fun intashr ((k Int) (a Int) (b Int) ) Int (ite (= (bitof k (- k 1) a) 0) (intlshr k a b) (intnot k (intlshr k (intnot k a) b))))
(define-fun intconcat ((k Int) (m Int) (a Int) (b Int)) Int (+ (* a (pow2 m)) b))
(define-fun intextract ( (i Int) (j Int) (x Int)) Int (mod (div x (pow2 j)) (pow2 (+ (- i j) 1)) ))
(define-fun intrepeatonebit ((x Int) (i Int)) Int (ite (= x 0) 0 (intmax i))) ;x is assumed to be either zero or one here!
(define-fun intadd ((k Int) (a Int) (b Int) ) Int (intmodtotal k (+ a b) (pow2 k)))
(define-fun intmul ((k Int) (a Int) (b Int)) Int (intmodtotal k (* a b) (pow2 k)))
(define-fun intsub ((k Int) (a Int) (b Int)) Int (intadd k a (intneg k b)))
(define-fun unsigned_to_signed ((k Int) (x Int)) Int (- (* 2 (int_all_but_msb k x)) x))
(define-fun intslt ((k Int) (a Int) (b Int)) Bool (< (unsigned_to_signed k a) (unsigned_to_signed k b)) )
(define-fun intsgt ((k Int) (a Int) (b Int)) Bool (> (unsigned_to_signed k a) (unsigned_to_signed k b)) )
(define-fun intsle ((k Int) (a Int) (b Int)) Bool (<= (unsigned_to_signed k a) (unsigned_to_signed k b)) )
(define-fun intsge ((k Int) (a Int) (b Int)) Bool (>= (unsigned_to_signed k a) (unsigned_to_signed k b)) )
(define-fun pow2_base_cases () Bool (and (= (pow2 0) 1) (= (pow2 1) 2) (= (pow2 2) 4) (= (pow2 3) 8) ) )
(define-fun pow2_ind_def () Bool (and (= (pow2 0) 1) (forall ((i Int)) (=> (> i 0) (= (pow2 i) (* (pow2 (- i 1)) 2))) ) ))
(define-fun and_ind_def ((k Int)) Bool (forall ((a Int) (b Int)) (=> (and (> k 0) (in_range k a) (in_range k b)) (= (intand k a b) (+ (ite (> k 1) (intand (- k 1) (int_all_but_msb k a) (int_all_but_msb k b)) 0) (* (pow2 (- k 1)) (intand_helper (bitof k (- k 1) a) (bitof k (- k 1) b)))))) ) )
(define-fun or_ind_def ((k Int)) Bool (forall ((a Int) (b Int))  (=> (and (> k 0) (in_range k a) (in_range k b)) (= (intor k a b) (+ (ite (> k 1) (intor (- k 1) (int_all_but_msb k a) (int_all_but_msb k b)) 0) (* (pow2 (- k 1)) (intor_helper (bitof k (- k 1) a) (bitof k (- k 1) b)))))) ) )
(define-fun xor_ind_def ((k Int)) Bool (forall ((a Int) (b Int))  (=> (and (> k 0) (in_range k a) (in_range k b)) (= (intxor k a b) (+ (ite (> k 1) (intxor (- k 1) (int_all_but_msb k a) (int_all_but_msb k b)) 0) (* (pow2 (- k 1)) (intxor_helper (bitof k (- k 1) a) (bitof k (- k 1) b))))))  ))
;pow2 properties
(define-fun pow2_weak_monotinicity () Bool (forall ((i Int) (j Int)) (=> (and (>= i 0) (>= j 0) ) (=> (<= i j) (<= (pow2 i) (pow2 j))) )))
(define-fun pow2_strong_monotinicity () Bool (forall ((i Int) (j Int)) (=> (and (>= i 0) (>= j 0) ) (=> (< i j) (< (pow2 i) (pow2 j))) ) ) )
(define-fun pow2_modular_power () Bool (forall ((i Int) (j Int) (x Int)) (=> (and (>= i 0) (>= j 0) (>= x 0) (distinct (mod (* x (pow2 i)) (pow2 j)) 0)) (< i j) )  ) )
(define-fun pow2_never_even () Bool (forall ((k Int) (x Int)) (=> (and (>= k 1) (>= x 0)) (distinct (- (pow2 k) 1) (* 2 x)) )) )
(define-fun pow2_always_positive () Bool (forall ((k Int)) (=> (>= k 0) (>= (pow2 k) 1) )  ) )
(define-fun pow2_div_zero () Bool (forall ((k Int)) (=> (>= k 0) (= (div k (pow2 k)) 0 ) )  ) )
(define-fun pow2_properties () Bool (and pow2_base_cases pow2_weak_monotinicity pow2_strong_monotinicity pow2_modular_power pow2_never_even pow2_always_positive pow2_div_zero ) )
;and properties
(define-fun and_max1 ((k Int)) Bool (forall ((a Int)) (=> (and (> k 0) (in_range k a)) (= (intand k a (intmax k)) a)) ) )
(define-fun and_max2 ((k Int)) Bool (forall ((a Int)) (=> (and (> k 0) (in_range k a)) (= (intand k 0 a) 0   ))  ))
(define-fun and_ide ((k Int)) Bool (forall ((a Int)) (=> (and (> k 0) (in_range k a)) (= (intand k a a) a))  ))
(define-fun rule_of_contradiction ((k Int)) Bool (forall ((a Int))  (=> (and (> k 0) (in_range k a))  (= (intand k (intnot k a) a) 0 ))  ))
(define-fun and_sym ((k Int)) Bool (forall ((a Int) (b Int)) (=> (and (> k 0) (in_range k a) (in_range k b)) (= (intand k a b) (intand k b a))) ))
(define-fun and_difference1 ((k Int)) Bool (forall ((a Int) (b Int) (c Int)) (=> (and (distinct a b) (> k 0) (in_range k a) (in_range k b) (in_range k c) ) (or (distinct (intand k a c) b) (distinct (intand k b c) a)))  ))
(define-fun and_ranges ((k Int)) Bool (forall ((a Int) (b Int)) (=> (and (> k 0) (in_range k a ) (in_range k b ) ) (and (in_range k (intand k a b)) (<= (intand k a b) a) (<= (intand k a b) b) ) ) ))
(define-fun and_properties ((k Int)) Bool (and (and_max1 k) (and_max2 k) (and_ide k) (rule_of_contradiction k) (and_sym k) (and_difference1 k) (and_ranges k) ))
;or properties
(define-fun or_max1 ((k Int)) Bool (forall ((a Int)) (=> (and (> k 0) (in_range k a)) (= (intor k a (intmax k)) (intmax k)))  ))
(define-fun or_max2 ((k Int)) Bool (forall ((a Int)) (=> (and (> k 0) (in_range k a)) (= (intor k a 0) a))  ))
(define-fun or_ide ((k Int)) Bool (forall ((a Int)) (=> (and (> k 0) (in_range k a)) (= (intor k a a) a))  ))
(define-fun excluded_middle ((k Int)) Bool (forall ((a Int)) (=> (and (> k 0) (in_range k a)) (and (= (intor k (intnot k a) a) (intmax k)) (= (intor k a (intnot k a)) (intmax k))  )) ))
(define-fun or_difference1 ((k Int)) Bool (forall ((a Int) (b Int) (c Int)) (=> (and (distinct a b) (> k 0) (in_range k a) (in_range k b) (in_range k c) ) (or (distinct (intor k a c) b) (distinct (intor k b c) a)))  ))
(define-fun or_sym ((k Int)) Bool (forall ((a Int) (b Int)) (=> (and (> k 0) (in_range k a) (in_range k b)) (= (intor k a b) (intor k b a))) ))
(define-fun or_ranges ((k Int)) Bool (forall ((a Int) (b Int)) (=> (and (> k 0) (in_range k a) (in_range k b) ) (and (in_range k (intor k a b)) (>= (intor k a b) a) (>= (intor k a b) b) ) )  ))
(define-fun or_properties ((k Int)) Bool (and (or_max1 k) (or_max2 k) (or_ide k) (excluded_middle k) (or_sym k) (or_difference1 k) (or_ranges k) ))
;xor properties
(define-fun xor_ide ((k Int)) Bool (forall ((a Int)) (=> (and (> k 0) (in_range k a) ) (= (intxor k a a) 0))  ))
(define-fun xor_flip ((k Int)) Bool (forall ((a Int)) (=> (and (> k 0) (in_range k a)) (= (intxor k a (intnot k a)) (intmax k))) ) )
(define-fun xor_sym ((k Int)) Bool (forall ((a Int) (b Int)) (=> (and (> k 0) (in_range k a) (in_range k b)) (= (intxor k a b) (intxor k b a))) ))
(define-fun xor_ranges ((k Int)) Bool (forall ((a Int) (b Int)) (=> (and (> k 0) (in_range k a) (in_range k b) ) (in_range k (intxor k a b)) )  ))
(define-fun xor_properties ((k Int)) Bool (and (xor_ide k) (xor_flip k) (xor_sym k) (xor_ranges k) ))
;combined axioms
(define-fun pow2_ax () Bool (and pow2_ind_def pow2_properties))
(define-fun and_ax ((k Int)) Bool (and (and_ind_def k) (and_properties k)))
(define-fun or_ax ((k Int)) Bool (and (or_ind_def k) (or_properties k)))
(define-fun xor_ax ((k Int)) Bool (and (xor_ind_def k) (xor_properties k)))
"""


AXIOMS = """; axioms
(assert pow2_ax)
(assert (and_ax k))
(assert (or_ax k))
(assert (xor_ax k))

"""


DROP_LINE_PATTERNS = (
    "set.logic",
    "set.option",
    "check-sat",
    "exit",
)

REPLACEMENTS = [
    (r"#b0000", "0"),
    (r"#b0001", "1"),
    (r"#b0010", "k"),
    (r"#b1111", "(intmax k)"),
    (r"#b0111", "(intmaxs k)"),
    (r"#b1000", "(intmins k)"),
    (r"\(_ bv0 k\)", "0"),
    (r"\(_ bv1 k\)", "1"),
    (r"\(_ bvk k\)", "k"),
    (r"\(check-sat\)", ""),
    (r"\(exit\)", ""),
    (r"\(_ BitVec k\)", "Int"),
    (r"\(_ bv([0-9][0-9]*) [0-9][0-9]*\)", r"\1"),
    (r"\(bvneg", "(intneg k"),
    (r"\(bvnot", "(intnot k"),
    (r"\(bvadd", "(intadd k"),
    (r"\(bvsub", "(intsub k"),
    (r"\(bvshl", "(intshl k"),
    (r"\(bvlshr", "(intlshr k"),
    (r"\(bvashr", "(intashr k"),
    (r"\(bvmul", "(intmul k"),
    (r"\(bvand", "(intand k"),
    (r"\(bvor", "(intor k"),
    (r"\(bvxor", "(intxor k"),
    (r"\(bvudiv", "(intudivtotal k"),
    (r"\(bvurem", "(intmodtotal k"),
    (r"\(bvult", "(<"),
    (r"\(bvule", "(<="),
    (r"\(bvugt", "(>"),
    (r"\(bvuge", "(>="),
    (r"\(bvslt", "(intslt k"),
    (r"\(bvsgt", "(intsgt k"),
    (r"\(bvsle", "(intsle k"),
    (r"\(bvsge", "(intsge k"),
    (r"\(bvule", "(<="),
]

DROP_BOUNDS_PATTERN = re.compile(r"0.k.....k..pow2.k")


def translated_path(benchmark_path: Path, override: str | None) -> Path:
    if override:
        return Path(override).expanduser()
    tmp_dir = Path(tempfile.gettempdir())
    sanitized = benchmark_path.as_posix().replace("/", "_")
    return tmp_dir / f"{sanitized}.bwind.smt2"


def generate_bounds(lines: Iterable[str], pattern: str) -> List[str]:
    regex = re.compile(pattern)
    bounds = []
    for line in lines:
        match = regex.match(line)
        if match:
            name = match.group(1)
            bounds.append(f"(assert (and (<= 0 {name}) (< {name} (pow2 k))))")
    return bounds


def translate_lines(lines: Iterable[str]) -> List[str]:
    translated = []
    for raw_line in lines:
        if any(tag in raw_line for tag in DROP_LINE_PATTERNS):
            continue
        line = raw_line
        for pattern, repl in REPLACEMENTS:
            line = re.sub(pattern, repl, line)
        translated.append(line)
    return translated


def write_translated_file(target: Path, translated_lines: Iterable[str], fun_bounds: Iterable[str], const_bounds: Iterable[str]) -> None:
    with target.open("w", encoding="utf-8") as f:
        f.write(TEMPLATE)
        for line in translated_lines:
            f.write(line)
        f.write("; bounds\n")
        for bound in fun_bounds:
            if not DROP_BOUNDS_PATTERN.search(bound):
                f.write(f"{bound}\n")
        for bound in const_bounds:
            if not DROP_BOUNDS_PATTERN.search(bound):
                f.write(f"{bound}\n")
        f.write(AXIOMS)
        f.write("(assert (> k 0))\n")
        f.write("(check-sat)\n")


def run_cvc5(cvc5_bin: Path, benchmark_path: Path) -> int:
    result = subprocess.run(
        [str(cvc5_bin), "--full-saturate-quant", "--nl-ext-tplanes", str(benchmark_path)],
        check=False,
    )
    return result.returncode


def main() -> None:
    parser = argparse.ArgumentParser(description="Translate a bit-vector benchmark to integer theory axioms and solve with cvc5.")
    parser.add_argument("cvc5_bin", type=str, help="Path to cvc5 binary")
    parser.add_argument("benchmark", type=str, help="Path to the input SMT2 benchmark")
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional output path for translated benchmark (defaults to /tmp/<sanitized>.bwind.smt2)",
    )
    args = parser.parse_args()

    cvc5_bin = Path(args.cvc5_bin)
    benchmark_path = Path(args.benchmark)

    if not benchmark_path.is_file():
        raise FileNotFoundError(f"Benchmark not found: {benchmark_path}")

    translated_file = translated_path(benchmark_path, args.out)
    lines = benchmark_path.read_text(encoding="utf-8").splitlines(keepends=True)

    fun_bounds = generate_bounds(lines, r"\(declare-fun\s+(\S+)")
    const_bounds = generate_bounds(lines, r"\(declare-const\s+(\S+)")
    translated_body = translate_lines(lines)

    write_translated_file(translated_file, translated_body, fun_bounds, const_bounds)

    ret = run_cvc5(cvc5_bin, translated_file)
    raise SystemExit(ret)


if __name__ == "__main__":
    main()
