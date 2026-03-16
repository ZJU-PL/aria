; arrow_hurwicz.txt converted to CHC format
; Original: Conditional discrete-time system
; State: x, y
; Initial: x ∈ [0.0, 1.5], y ∈ [0.375, 1.375]
; Transition: if y + 0.25*(-x + 1) <= 0 then {y' = 0, x' = x - 0.5*(x - y)}
;             else {x' = x - 0.5*(x - y), y' = y + 0.25*(-x + 1)}
; Safety: Verify state variables stay within bounds

(set-logic HORN)

(declare-fun inv ((_ FloatingPoint 8 24) (_ FloatingPoint 8 24)) Bool)

; Initial state: x ∈ [0.0, 1.5], y ∈ [0.375, 1.375]
(assert (forall ((x (_ FloatingPoint 8 24)) (y (_ FloatingPoint 8 24)))
         (=> (and (fp.geq x ((_ to_fp 8 24) RNE 0.0))
                  (fp.leq x ((_ to_fp 8 24) RNE 1.5))
                  (fp.geq y ((_ to_fp 8 24) RNE 0.375))
                  (fp.leq y ((_ to_fp 8 24) RNE 1.375)))
             (inv x y))))

; Transition relation with conditional logic
; If y + 0.25*(-x + 1) <= 0: y' = 0, x' = x - 0.5*(x - y)
; Else: x' = x - 0.5*(x - y), y' = y + 0.25*(-x + 1)
(assert (forall ((x (_ FloatingPoint 8 24)) (y (_ FloatingPoint 8 24))
                 (x! (_ FloatingPoint 8 24)) (y! (_ FloatingPoint 8 24)))
         (=> (and (inv x y)
                  (ite (fp.leq (fp.add RNE y (fp.mul RNE ((_ to_fp 8 24) RNE 0.25) (fp.sub RNE ((_ to_fp 8 24) RNE 1.0) x)))
                              ((_ to_fp 8 24) RNE 0.0))
                       (and (fp.eq y! ((_ to_fp 8 24) RNE 0.0))
                            (fp.eq x! (fp.sub RNE x (fp.mul RNE ((_ to_fp 8 24) RNE 0.5) (fp.sub RNE x y)))))
                       (and (fp.eq x! (fp.sub RNE x (fp.mul RNE ((_ to_fp 8 24) RNE 0.5) (fp.sub RNE x y))))
                            (fp.eq y! (fp.add RNE y (fp.mul RNE ((_ to_fp 8 24) RNE 0.25) (fp.sub RNE ((_ to_fp 8 24) RNE 1.0) x))))))
             (inv x! y!))))

; Safety property: state variables should stay within reasonable bounds
(assert (forall ((x (_ FloatingPoint 8 24)) (y (_ FloatingPoint 8 24)))
         (=> (inv x y)
             (and (fp.geq x ((_ to_fp 8 24) RNE -2.0))
                  (fp.leq x ((_ to_fp 8 24) RNE 2.0))
                  (fp.geq y ((_ to_fp 8 24) RNE -2.0))
                  (fp.leq y ((_ to_fp 8 24) RNE 2.0))))))

(check-sat)

