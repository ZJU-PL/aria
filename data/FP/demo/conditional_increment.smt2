;Initial: x ∈ [0.0, 3.0]
;Transition: if x < 2.0 then x' = x + 1.0 else x' = x - 0.5
;Safety: x ∈ [0.0, 5.0]
;Expected: SAT (conditional logic keeps within bounds)

(set-logic HORN)
(declare-fun inv ((_ FloatingPoint 8 24)) Bool)
(assert (forall ((x (_ FloatingPoint 8 24))) 
       (=> (and (fp.geq x ((_ to_fp 8 24) RNE 0.0)) (fp.leq x ((_ to_fp 8 24) RNE 3.0))) (inv x))))
(assert (forall ((x (_ FloatingPoint 8 24)) (x! (_ FloatingPoint 8 24))) 
       (=> (and (inv x) (ite (fp.lt x ((_ to_fp 8 24) RNE 2.0)) (fp.eq x! (fp.add RNE x ((_ to_fp 8 24) RNE 1.0))) (fp.eq x! (fp.sub RNE x ((_ to_fp 8 24) RNE 0.5))))) (inv x!))))
(assert (forall ((x (_ FloatingPoint 8 24))) 
       (=> (inv x) (and (fp.geq x ((_ to_fp 8 24) RNE 0.0)) (fp.leq x ((_ to_fp 8 24) RNE 5.0))))))
(check-sat)
