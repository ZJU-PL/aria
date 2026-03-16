;Initial: x ∈ [1.0, 5.0]
;Transition: x' = x / 2.0 (halving)
;Safety: x ≥ 0.1
;Expected: SAT (division keeps values positive)

(set-logic HORN)
(declare-fun inv ((_ FloatingPoint 8 24)) Bool)
(assert (forall ((x (_ FloatingPoint 8 24))) 
       (=> (and (fp.geq x ((_ to_fp 8 24) RNE 1.0)) (fp.leq x ((_ to_fp 8 24) RNE 5.0))) (inv x))))
(assert (forall ((x (_ FloatingPoint 8 24)) (x! (_ FloatingPoint 8 24))) 
       (=> (and (inv x) (fp.eq x! (fp.div RNE x ((_ to_fp 8 24) RNE 2.0)))) (inv x!))))
(assert (forall ((x (_ FloatingPoint 8 24))) 
       (=> (inv x) (fp.geq x ((_ to_fp 8 24) RNE 0.1)))))
(check-sat)
