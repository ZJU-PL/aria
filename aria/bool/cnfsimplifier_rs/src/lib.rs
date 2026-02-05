//! CNF simplifier - Rust implementation with Python bindings.
//! Preserves satisfiability; clauses are lists of non-zero integers (literals).

use pyo3::prelude::*;
use std::collections::HashSet;

type Literal = i32;
type Clause = Vec<Literal>;
type Cnf = Vec<Clause>;

fn clause_set(c: &[Literal]) -> HashSet<Literal> {
    c.iter().copied().filter(|&x| x != 0).collect()
}

fn is_tautology(c: &[Literal]) -> bool {
    let s = clause_set(c);
    for &lit in &s {
        if s.contains(&(-lit)) {
            return true;
        }
    }
    false
}

fn is_sub_clause_of(sub: &[Literal], sup: &[Literal]) -> bool {
    let sup_set = clause_set(sup);
    sub.iter().all(|&lit| lit == 0 || sup_set.contains(&lit))
}

fn resolvent(c1: &[Literal], c2: &[Literal], lit: Literal) -> Option<Clause> {
    if lit == 0 || !c1.iter().any(|&x| x == lit) || !c2.iter().any(|&x| x == -lit) {
        return None;
    }
    let s1: HashSet<Literal> = c1.iter().copied().filter(|&x| x != 0 && x != lit).collect();
    let s2: HashSet<Literal> = c2.iter().copied().filter(|&x| x != 0 && x != -lit).collect();
    let mut res: Clause = s1.union(&s2).copied().collect();
    res.sort_by_key(|x| x.abs());
    Some(res)
}

/// Returns (blocking_clause_index, _) if this clause is blocked by some other clause in cnf.
fn get_blocking_clause_idx(clause: &[Literal], cnf: &[Clause], self_idx: usize) -> Option<usize> {
    for (idx, other) in cnf.iter().enumerate() {
        if idx == self_idx {
            continue;
        }
        for &lit in clause {
            if lit == 0 {
                continue;
            }
            if other.iter().any(|&x| x == -lit) {
                if let Some(res) = resolvent(clause, other, lit) {
                    if is_tautology(&res) {
                        return Some(idx);
                    }
                }
            }
        }
    }
    None
}

fn is_subsumed(clause: &[Literal], cnf: &[Clause], self_idx: usize) -> bool {
    for (idx, other) in cnf.iter().enumerate() {
        if idx != self_idx && is_sub_clause_of(other, clause) {
            return true;
        }
    }
    false
}

/// Hidden Literal Addition: add literals from binary clauses in cnf that contain a literal from clause.
fn hla(clause: &[Literal], cnf: &[Clause], self_idx: usize) -> Clause {
    let mut result: HashSet<Literal> = clause.iter().copied().filter(|&x| x != 0).collect();
    for &lit in clause {
        if lit == 0 {
            continue;
        }
        for (idx, other) in cnf.iter().enumerate() {
            if idx == self_idx || other.len() != 2 {
                continue;
            }
            let other_set = clause_set(other);
            if other_set.contains(&lit) {
                let other_lit = other.iter().find(|&&x| x != 0 && x != lit).copied();
                if let Some(l) = other_lit {
                    result.insert(l);
                }
            }
        }
    }
    let mut res: Clause = result.into_iter().collect();
    res.sort_by_key(|x| x.abs());
    res
}

/// Asymmetric Literal Addition: add literals from clauses that extend subsets of this clause.
fn ala(clause: &[Literal], cnf: &[Clause], self_idx: usize) -> Clause {
    let mut result: HashSet<Literal> = clause.iter().copied().filter(|&x| x != 0).collect();
    let lits: Vec<Literal> = clause.iter().copied().filter(|&x| x != 0).collect();
    // iterate over non-empty subsets of lits
    let n = lits.len();
    for mask in 1u64..(1 << n) {
        let subset: Vec<Literal> = (0..n)
            .filter(|i| (mask >> i) & 1 != 0)
            .map(|i| lits[i])
            .collect();
        let subset_set: HashSet<Literal> = subset.iter().copied().collect();
        for (idx, other) in cnf.iter().enumerate() {
            if idx == self_idx {
                continue;
            }
            let other_set = clause_set(other);
            if other_set.len() == subset.len() + 1 && subset.iter().all(|&l| other_set.contains(&l)) {
                let diff: Literal = other_set
                    .iter()
                    .copied()
                    .find(|lit| !subset_set.contains(&lit))
                    .unwrap();
                result.insert(diff);
            }
        }
    }
    let mut res: Clause = result.into_iter().collect();
    res.sort_by_key(|x| x.abs());
    res
}

fn tautology_elimination(cnf: Cnf) -> Cnf {
    cnf.into_iter().filter(|c| !is_tautology(c)).collect()
}

fn subsumption_elimination(mut cnf: Cnf) -> Cnf {
    loop {
        let prev_len = cnf.len();
        let mut i = 0;
        while i < cnf.len() {
            if is_subsumed(&cnf[i], &cnf, i) {
                cnf.remove(i);
            } else {
                i += 1;
            }
        }
        if cnf.len() == prev_len {
            break;
        }
    }
    cnf
}

fn blocked_clause_elimination(mut cnf: Cnf) -> Cnf {
    loop {
        let prev_len = cnf.len();
        let mut i = 0;
        while i < cnf.len() {
            if let Some(j) = get_blocking_clause_idx(&cnf[i], &cnf, i) {
                if cnf[j].len() < cnf[i].len() {
                    cnf.remove(j);
                    if j < i {
                        i -= 1;
                    }
                } else {
                    cnf.remove(i);
                    i -= 1;
                }
            }
            i += 1;
        }
        if cnf.len() == prev_len {
            break;
        }
    }
    cnf
}

fn hidden_tautology_elimination(mut cnf: Cnf) -> Cnf {
    loop {
        let prev_len = cnf.len();
        let mut i = 0;
        while i < cnf.len() {
            let hla_c = hla(&cnf[i], &cnf, i);
            if is_tautology(&hla_c) {
                cnf.remove(i);
            } else {
                i += 1;
            }
        }
        if cnf.len() == prev_len {
            break;
        }
    }
    cnf
}

fn hidden_subsumption_elimination(mut cnf: Cnf) -> Cnf {
    loop {
        let prev_len = cnf.len();
        let mut i = 0;
        while i < cnf.len() {
            let hla_c = hla(&cnf[i], &cnf, i);
            let subsumed = cnf.iter().enumerate().any(|(idx, other)| {
                idx != i && is_sub_clause_of(other, &hla_c)
            });
            if subsumed {
                cnf.remove(i);
            } else {
                i += 1;
            }
        }
        if cnf.len() == prev_len {
            break;
        }
    }
    cnf
}

fn hidden_blocked_clause_elimination(mut cnf: Cnf) -> Cnf {
    loop {
        let prev_len = cnf.len();
        let mut i = 0;
        while i < cnf.len() {
            let hla_c = hla(&cnf[i], &cnf, i);
            // Check if hla_c is blocked in cnf (blocking check: exists other clause D in cnf such that resolvent is tautology)
            let mut blocked = false;
            for (idx, other) in cnf.iter().enumerate() {
                if idx == i {
                    continue;
                }
                for &lit in &hla_c {
                    if other.iter().any(|&x| x == -lit) {
                        if let Some(res) = resolvent(&hla_c, other, lit) {
                            if is_tautology(&res) {
                                blocked = true;
                                break;
                            }
                        }
                    }
                }
                if blocked {
                    break;
                }
            }
            if blocked {
                cnf.remove(i);
            } else {
                i += 1;
            }
        }
        if cnf.len() == prev_len {
            break;
        }
    }
    cnf
}

fn asymmetric_tautology_elimination(mut cnf: Cnf) -> Cnf {
    loop {
        let prev_len = cnf.len();
        let mut i = 0;
        while i < cnf.len() {
            let ala_c = ala(&cnf[i], &cnf, i);
            if is_tautology(&ala_c) {
                cnf.remove(i);
            } else {
                i += 1;
            }
        }
        if cnf.len() == prev_len {
            break;
        }
    }
    cnf
}

fn asymmetric_subsumption_elimination(mut cnf: Cnf) -> Cnf {
    loop {
        let prev_len = cnf.len();
        let mut i = 0;
        while i < cnf.len() {
            let ala_c = ala(&cnf[i], &cnf, i);
            let subsumed = cnf.iter().enumerate().any(|(idx, other)| {
                idx != i && is_sub_clause_of(other, &ala_c)
            });
            if subsumed {
                cnf.remove(i);
            } else {
                i += 1;
            }
        }
        if cnf.len() == prev_len {
            break;
        }
    }
    cnf
}

fn asymmetric_blocked_clause_elimination(mut cnf: Cnf) -> Cnf {
    loop {
        let prev_len = cnf.len();
        let mut i = 0;
        while i < cnf.len() {
            let ala_c = ala(&cnf[i], &cnf, i);
            let mut blocked = false;
            for (idx, other) in cnf.iter().enumerate() {
                if idx == i {
                    continue;
                }
                for &lit in &ala_c {
                    if other.iter().any(|&x| x == -lit) {
                        if let Some(res) = resolvent(&ala_c, other, lit) {
                            if is_tautology(&res) {
                                blocked = true;
                                break;
                            }
                        }
                    }
                }
                if blocked {
                    break;
                }
            }
            if blocked {
                cnf.remove(i);
            } else {
                i += 1;
            }
        }
        if cnf.len() == prev_len {
            break;
        }
    }
    cnf
}

fn filter_zeros(clauses: &[Vec<i32>]) -> Vec<Vec<i32>> {
    clauses
        .iter()
        .map(|c| c.iter().copied().filter(|&x| x != 0).collect())
        .collect()
}

// --------------- Python bindings ---------------

#[pyfunction]
fn simplify_numeric_clauses(clauses: Vec<Vec<i32>>) -> PyResult<Vec<Vec<i32>>> {
    let cnf = filter_zeros(&clauses);
    let out = subsumption_elimination(cnf);
    Ok(out)
}

#[pyfunction]
fn cnf_tautology_elimination(clauses: Vec<Vec<i32>>) -> PyResult<Vec<Vec<i32>>> {
    let cnf = filter_zeros(&clauses);
    Ok(tautology_elimination(cnf))
}

#[pyfunction]
fn cnf_subsumption_elimination(clauses: Vec<Vec<i32>>) -> PyResult<Vec<Vec<i32>>> {
    let cnf = filter_zeros(&clauses);
    Ok(subsumption_elimination(cnf))
}

#[pyfunction]
fn cnf_blocked_clause_elimination(clauses: Vec<Vec<i32>>) -> PyResult<Vec<Vec<i32>>> {
    let cnf = filter_zeros(&clauses);
    Ok(blocked_clause_elimination(cnf))
}

#[pyfunction]
fn cnf_hidden_tautology_elimination(clauses: Vec<Vec<i32>>) -> PyResult<Vec<Vec<i32>>> {
    let cnf = filter_zeros(&clauses);
    Ok(hidden_tautology_elimination(cnf))
}

#[pyfunction]
fn cnf_hidden_subsumption_elimination(clauses: Vec<Vec<i32>>) -> PyResult<Vec<Vec<i32>>> {
    let cnf = filter_zeros(&clauses);
    Ok(hidden_subsumption_elimination(cnf))
}

#[pyfunction]
fn cnf_hidden_blocked_clause_elimination(clauses: Vec<Vec<i32>>) -> PyResult<Vec<Vec<i32>>> {
    let cnf = filter_zeros(&clauses);
    Ok(hidden_blocked_clause_elimination(cnf))
}

#[pyfunction]
fn cnf_asymmetric_tautology_elimination(clauses: Vec<Vec<i32>>) -> PyResult<Vec<Vec<i32>>> {
    let cnf = filter_zeros(&clauses);
    Ok(asymmetric_tautology_elimination(cnf))
}

#[pyfunction]
fn cnf_asymmetric_subsumption_elimination(clauses: Vec<Vec<i32>>) -> PyResult<Vec<Vec<i32>>> {
    let cnf = filter_zeros(&clauses);
    Ok(asymmetric_subsumption_elimination(cnf))
}

#[pyfunction]
fn cnf_asymmetric_blocked_clause_elimination(clauses: Vec<Vec<i32>>) -> PyResult<Vec<Vec<i32>>> {
    let cnf = filter_zeros(&clauses);
    Ok(asymmetric_blocked_clause_elimination(cnf))
}

#[pymodule]
fn cnfsimplifier_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(simplify_numeric_clauses, m)?)?;
    m.add_function(wrap_pyfunction!(cnf_tautology_elimination, m)?)?;
    m.add_function(wrap_pyfunction!(cnf_subsumption_elimination, m)?)?;
    m.add_function(wrap_pyfunction!(cnf_blocked_clause_elimination, m)?)?;
    m.add_function(wrap_pyfunction!(cnf_hidden_tautology_elimination, m)?)?;
    m.add_function(wrap_pyfunction!(cnf_hidden_subsumption_elimination, m)?)?;
    m.add_function(wrap_pyfunction!(cnf_hidden_blocked_clause_elimination, m)?)?;
    m.add_function(wrap_pyfunction!(cnf_asymmetric_tautology_elimination, m)?)?;
    m.add_function(wrap_pyfunction!(cnf_asymmetric_subsumption_elimination, m)?)?;
    m.add_function(wrap_pyfunction!(cnf_asymmetric_blocked_clause_elimination, m)?)?;
    Ok(())
}
