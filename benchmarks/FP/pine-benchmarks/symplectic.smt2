; symplectic.txt converted to CHC format
; Original: Symplectic system
; State: x, v
; Initial: x, v ∈ [0.0, 1.0]
; Transition: x' = 0.95*x + 0.09975*v
;            v' = -0.1*x + 0.95*v
; Safety: Verify state variables stay within [0.0, 1.0]

(set-logic HORN)

(declare-fun inv ((_ FloatingPoint 8 24) (_ FloatingPoint 8 24)) Bool)

; Initial state: x, v ∈ [0.0, 1.0]
(assert (forall ((x (_ FloatingPoint 8 24)) (v (_ FloatingPoint 8 24)))
         (=> (and (fp.geq x ((_ to_fp 8 24) RNE 0.0))
                  (fp.leq x ((_ to_fp 8 24) RNE 1.0))
                  (fp.geq v ((_ to_fp 8 24) RNE 0.0))
                  (fp.leq v ((_ to_fp 8 24) RNE 1.0)))
             (inv x v))))

; Transition relation: x' = 0.95*x + 0.09975*v, v' = -0.1*x + 0.95*v
(assert (forall ((x (_ FloatingPoint 8 24)) (v (_ FloatingPoint 8 24))
                 (x! (_ FloatingPoint 8 24)) (v! (_ FloatingPoint 8 24)))
         (=> (and (inv x v)
                  (fp.eq x! (fp.add RNE (fp.mul RNE ((_ to_fp 8 24) RNE 0.95) x)
                                         (fp.mul RNE ((_ to_fp 8 24) RNE 0.09975) v)))
                  (fp.eq v! (fp.add RNE (fp.mul RNE ((_ to_fp 8 24) RNE -0.1) x)
                                         (fp.mul RNE ((_ to_fp 8 24) RNE 0.95) v))))
             (inv x! v!))))

; Safety property: state variables should stay within [0.0, 1.0]
(assert (forall ((x (_ FloatingPoint 8 24)) (v (_ FloatingPoint 8 24)))
         (=> (inv x v)
             (and (fp.geq x ((_ to_fp 8 24) RNE 0.0))
                  (fp.leq x ((_ to_fp 8 24) RNE 1.0))
                  (fp.geq v ((_ to_fp 8 24) RNE 0.0))
                  (fp.leq v ((_ to_fp 8 24) RNE 1.0))))))

(check-sat)

