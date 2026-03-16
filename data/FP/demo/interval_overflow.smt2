; Initial: x ∈ [0.0, 1.0]
; Transition: x' = x * 2.0 (doubling)
; Safety: x ≤ 2.0
; Expected: UNSAT (will overflow)

(set-logic HORN)
(declare-fun inv ((_ FloatingPoint 8 24)) Bool)
(assert (forall ((x (_ FloatingPoint 8 24))) 
       (=> (and (fp.geq x ((_ to_fp 8 24) RNE 0.0)) (fp.leq x ((_ to_fp 8 24) RNE 1.0))) (inv x))))
(assert (forall ((x (_ FloatingPoint 8 24)) (x! (_ FloatingPoint 8 24))) 
       (=> (and (inv x) (fp.eq x! (fp.mul RNE x ((_ to_fp 8 24) RNE 2.0)))) (inv x!))))
(assert (forall ((x (_ FloatingPoint 8 24))) 
       (=> (inv x) (fp.leq x ((_ to_fp 8 24) RNE 2.0)))))
(check-sat)
