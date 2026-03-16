; nonlin_example3.txt converted to CHC format
; Original: Nonlinear discrete-time system
; State: x, y
; Initial: x, y ∈ [-1.5, 0.6]
; Transition: x' = x + 0.01*(-x + y*y)
;            y' = y + 0.01*(-2.0*y + 3.0*x*x)
; Safety: Verify state variables stay within [-1.5, 0.6]

(set-logic HORN)

(declare-fun inv ((_ FloatingPoint 8 24) (_ FloatingPoint 8 24)) Bool)

; Initial state: x, y ∈ [-1.5, 0.6]
(assert (forall ((x (_ FloatingPoint 8 24)) (y (_ FloatingPoint 8 24)))
         (=> (and (fp.geq x ((_ to_fp 8 24) RNE (- 1.5)))
                  (fp.leq x ((_ to_fp 8 24) RNE 0.6))
                  (fp.geq y ((_ to_fp 8 24) RNE (- 1.5)))
                  (fp.leq y ((_ to_fp 8 24) RNE 0.6)))
             (inv x y))))

; Transition relation: x' = x + 0.01*(-x + y*y), y' = y + 0.01*(-2.0*y + 3.0*x*x)
(assert (forall ((x (_ FloatingPoint 8 24)) (y (_ FloatingPoint 8 24))
                 (x! (_ FloatingPoint 8 24)) (y! (_ FloatingPoint 8 24)))
         (=> (and (inv x y)
                  (fp.eq x! (fp.add RNE x (fp.mul RNE ((_ to_fp 8 24) RNE 0.01)
                                         (fp.add RNE (fp.sub RNE ((_ to_fp 8 24) RNE 0.0) x)
                                                (fp.mul RNE y y)))))
                  (fp.eq y! (fp.add RNE y (fp.mul RNE ((_ to_fp 8 24) RNE 0.01)
                                         (fp.add RNE (fp.mul RNE ((_ to_fp 8 24) RNE -2.0) y)
                                                (fp.mul RNE ((_ to_fp 8 24) RNE 3.0) (fp.mul RNE x x))))))
             (inv x! y!))))

; Safety property: state variables should stay within [-1.5, 0.6]
(assert (forall ((x (_ FloatingPoint 8 24)) (y (_ FloatingPoint 8 24)))
         (=> (inv x y)
             (and (fp.geq x ((_ to_fp 8 24) RNE (- 1.5)))
                  (fp.leq x ((_ to_fp 8 24) RNE 0.6))
                  (fp.geq y ((_ to_fp 8 24) RNE (- 1.5)))
                  (fp.leq y ((_ to_fp 8 24) RNE 0.6))))))

(check-sat)

