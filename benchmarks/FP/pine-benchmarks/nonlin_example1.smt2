; nonlin_example1.txt converted to CHC format
; Original: Nonlinear discrete-time system
; State: x, y
; Initial: x, y ∈ [0.0, 0.1]
; Transition: x' = x + 0.01*(-2*x - 3*y + x*x)
;            y' = y + 0.01*(x + y)
; Safety: Verify state variables stay within [0.0, 0.1]

(set-logic HORN)

(declare-fun inv ((_ FloatingPoint 8 24) (_ FloatingPoint 8 24)) Bool)

; Initial state: x, y ∈ [0.0, 0.1]
(assert (forall ((x (_ FloatingPoint 8 24)) (y (_ FloatingPoint 8 24)))
         (=> (and (fp.geq x ((_ to_fp 8 24) RNE 0.0))
                  (fp.leq x ((_ to_fp 8 24) RNE 0.1))
                  (fp.geq y ((_ to_fp 8 24) RNE 0.0))
                  (fp.leq y ((_ to_fp 8 24) RNE 0.1)))
             (inv x y))))

; Transition relation: x' = x + 0.01*(-2*x - 3*y + x*x), y' = y + 0.01*(x + y)
(assert (forall ((x (_ FloatingPoint 8 24)) (y (_ FloatingPoint 8 24))
                 (x! (_ FloatingPoint 8 24)) (y! (_ FloatingPoint 8 24)))
         (=> (and (inv x y)
                  (fp.eq x! (fp.add RNE x (fp.mul RNE ((_ to_fp 8 24) RNE 0.01)
                                         (fp.add RNE (fp.add RNE (fp.mul RNE ((_ to_fp 8 24) RNE -2.0) x)
                                                                 (fp.mul RNE ((_ to_fp 8 24) RNE -3.0) y))
                                                (fp.mul RNE x x)))))
                  (fp.eq y! (fp.add RNE y (fp.mul RNE ((_ to_fp 8 24) RNE 0.01)
                                         (fp.add RNE x y)))))
             (inv x! y!))))

; Safety property: state variables should stay within [0.0, 0.1]
(assert (forall ((x (_ FloatingPoint 8 24)) (y (_ FloatingPoint 8 24)))
         (=> (inv x y)
             (and (fp.geq x ((_ to_fp 8 24) RNE 0.0))
                  (fp.leq x ((_ to_fp 8 24) RNE 0.1))
                  (fp.geq y ((_ to_fp 8 24) RNE 0.0))
                  (fp.leq y ((_ to_fp 8 24) RNE 0.1))))))

(check-sat)

