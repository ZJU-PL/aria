; harmonic.txt converted to CHC format
; Original: Harmonic system
; State: x1, x2
; Initial: x1, x2 ∈ [0.0, 1.0]
; Transition: x1' = x1 + 0.01*x2
;            x2' = -0.01*x1 + 0.99*x2
; Safety: Verify state variables stay within [0.0, 1.0]

(set-logic HORN)

(declare-fun inv ((_ FloatingPoint 8 24) (_ FloatingPoint 8 24)) Bool)

; Initial state: x1, x2 ∈ [0.0, 1.0]
(assert (forall ((x1 (_ FloatingPoint 8 24)) (x2 (_ FloatingPoint 8 24)))
         (=> (and (fp.geq x1 ((_ to_fp 8 24) RNE 0.0))
                  (fp.leq x1 ((_ to_fp 8 24) RNE 1.0))
                  (fp.geq x2 ((_ to_fp 8 24) RNE 0.0))
                  (fp.leq x2 ((_ to_fp 8 24) RNE 1.0)))
             (inv x1 x2))))

; Transition relation: x1' = x1 + 0.01*x2, x2' = -0.01*x1 + 0.99*x2
(assert (forall ((x1 (_ FloatingPoint 8 24)) (x2 (_ FloatingPoint 8 24))
                 (x1! (_ FloatingPoint 8 24)) (x2! (_ FloatingPoint 8 24)))
         (=> (and (inv x1 x2)
                  (fp.eq x1! (fp.add RNE x1 (fp.mul RNE ((_ to_fp 8 24) RNE 0.01) x2)))
                  (fp.eq x2! (fp.add RNE (fp.mul RNE ((_ to_fp 8 24) RNE -0.01) x1)
                                         (fp.mul RNE ((_ to_fp 8 24) RNE 0.99) x2))))
             (inv x1! x2!))))

; Safety property: state variables should stay within [0.0, 1.0]
(assert (forall ((x1 (_ FloatingPoint 8 24)) (x2 (_ FloatingPoint 8 24)))
         (=> (inv x1 x2)
             (and (fp.geq x1 ((_ to_fp 8 24) RNE 0.0))
                  (fp.leq x1 ((_ to_fp 8 24) RNE 1.0))
                  (fp.geq x2 ((_ to_fp 8 24) RNE 0.0))
                  (fp.leq x2 ((_ to_fp 8 24) RNE 1.0))))))

(check-sat)

