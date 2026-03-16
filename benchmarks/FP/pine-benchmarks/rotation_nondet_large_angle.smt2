; rotation_nondet_large_angle.txt converted to CHC format
; Original: Rotation system with large angle and nondeterministic input
; State: x, y
; Input: th ∈ [-0.5, 0.5]
; Initial: x, y ∈ [-1.0, 1.0], th ∈ [-0.5, 0.5]
; Transition: x' = x*(1 - 0.5*th*th) - y*(th - th*th*th/6.0)
;            y' = x*(th - th*th*th/6.0) + y*(1 - 0.5*th*th)
; Safety: Verify state variables stay within [-1.0, 1.0]

(set-logic HORN)

(declare-fun inv ((_ FloatingPoint 8 24) (_ FloatingPoint 8 24) (_ FloatingPoint 8 24)) Bool)

; Initial state: x, y ∈ [-1.0, 1.0], th ∈ [-0.5, 0.5]
(assert (forall ((x (_ FloatingPoint 8 24)) (y (_ FloatingPoint 8 24)) (th (_ FloatingPoint 8 24)))
         (=> (and (fp.geq x ((_ to_fp 8 24) RNE (- 1.0)))
                  (fp.leq x ((_ to_fp 8 24) RNE 1.0))
                  (fp.geq y ((_ to_fp 8 24) RNE (- 1.0)))
                  (fp.leq y ((_ to_fp 8 24) RNE 1.0))
                  (fp.geq th ((_ to_fp 8 24) RNE (- 0.5)))
                  (fp.leq th ((_ to_fp 8 24) RNE 0.5)))
             (inv x y th))))

; Transition relation: x' = x*(1 - 0.5*th*th) - y*(th - th*th*th/6.0)
; y' = x*(th - th*th*th/6.0) + y*(1 - 0.5*th*th)
; th' can be any value in [-0.5, 0.5]
(assert (forall ((x (_ FloatingPoint 8 24)) (y (_ FloatingPoint 8 24)) (th (_ FloatingPoint 8 24))
                 (x! (_ FloatingPoint 8 24)) (y! (_ FloatingPoint 8 24)) (th! (_ FloatingPoint 8 24)))
         (=> (and (inv x y th)
                  (fp.geq th! ((_ to_fp 8 24) RNE (- 0.5)))
                  (fp.leq th! ((_ to_fp 8 24) RNE 0.5))
                  (fp.eq x! (fp.sub RNE (fp.mul RNE x (fp.sub RNE ((_ to_fp 8 24) RNE 1.0) (fp.mul RNE ((_ to_fp 8 24) RNE 0.5) (fp.mul RNE th th))))
                                         (fp.mul RNE y (fp.sub RNE th (fp.div RNE (fp.mul RNE th (fp.mul RNE th th)) ((_ to_fp 8 24) RNE 6.0))))))
                  (fp.eq y! (fp.add RNE (fp.mul RNE x (fp.sub RNE th (fp.div RNE (fp.mul RNE th (fp.mul RNE th th)) ((_ to_fp 8 24) RNE 6.0))))
                                         (fp.mul RNE y (fp.sub RNE ((_ to_fp 8 24) RNE 1.0) (fp.mul RNE ((_ to_fp 8 24) RNE 0.5) (fp.mul RNE th th)))))))
             (inv x! y! th!))))

; Safety property: state variables should stay within [-1.0, 1.0]
(assert (forall ((x (_ FloatingPoint 8 24)) (y (_ FloatingPoint 8 24)) (th (_ FloatingPoint 8 24)))
         (=> (inv x y th)
             (and (fp.geq x ((_ to_fp 8 24) RNE (- 1.0)))
                  (fp.leq x ((_ to_fp 8 24) RNE 1.0))
                  (fp.geq y ((_ to_fp 8 24) RNE (- 1.0)))
                  (fp.leq y ((_ to_fp 8 24) RNE 1.0))))))

(check-sat)

