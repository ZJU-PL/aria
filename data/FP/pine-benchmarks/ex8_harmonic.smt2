; ex8_harmonic.txt converted to CHC format
; Original: Linear discrete-time harmonic system
; State: x0, x1
; Initial: x0, x1 ∈ [0.0, 1.0]
; Transition: x0' = 0.95*x0 + 0.09975*x1
;            x1' = -0.1*x0 + 0.95*x1
; Safety: Verify state variables stay within [0.0, 1.0]

(set-logic HORN)

(declare-fun inv ((_ FloatingPoint 8 24) (_ FloatingPoint 8 24)) Bool)

; Initial state: x0, x1 ∈ [0.0, 1.0]
(assert (forall ((x0 (_ FloatingPoint 8 24)) (x1 (_ FloatingPoint 8 24)))
         (=> (and (fp.geq x0 ((_ to_fp 8 24) RNE 0.0))
                  (fp.leq x0 ((_ to_fp 8 24) RNE 1.0))
                  (fp.geq x1 ((_ to_fp 8 24) RNE 0.0))
                  (fp.leq x1 ((_ to_fp 8 24) RNE 1.0)))
             (inv x0 x1))))

; Transition relation: x0' = 0.95*x0 + 0.09975*x1, x1' = -0.1*x0 + 0.95*x1
(assert (forall ((x0 (_ FloatingPoint 8 24)) (x1 (_ FloatingPoint 8 24))
                 (x0! (_ FloatingPoint 8 24)) (x1! (_ FloatingPoint 8 24)))
         (=> (and (inv x0 x1)
                  (fp.eq x0! (fp.add RNE (fp.mul RNE ((_ to_fp 8 24) RNE 0.95) x0)
                                         (fp.mul RNE ((_ to_fp 8 24) RNE 0.09975) x1)))
                  (fp.eq x1! (fp.add RNE (fp.mul RNE ((_ to_fp 8 24) RNE -0.1) x0)
                                         (fp.mul RNE ((_ to_fp 8 24) RNE 0.95) x1))))
             (inv x0! x1!))))

; Safety property: state variables should stay within [0.0, 1.0]
(assert (forall ((x0 (_ FloatingPoint 8 24)) (x1 (_ FloatingPoint 8 24)))
         (=> (inv x0 x1)
             (and (fp.geq x0 ((_ to_fp 8 24) RNE 0.0))
                  (fp.leq x0 ((_ to_fp 8 24) RNE 1.0))
                  (fp.geq x1 ((_ to_fp 8 24) RNE 0.0))
                  (fp.leq x1 ((_ to_fp 8 24) RNE 1.0))))))

(check-sat)
