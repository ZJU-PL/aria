; filter_goubault.txt converted to CHC format
; Original: Filter system
; State: x, y
; Initial: x, y ∈ [0.0, 1.0]
; Transition: x' = 0.75*x - 0.125*y
;            y' = x
; Safety: Verify state variables stay within [0.0, 1.0]

(set-logic HORN)

(declare-fun inv ((_ FloatingPoint 8 24) (_ FloatingPoint 8 24)) Bool)

; Initial state: x, y ∈ [0.0, 1.0]
(assert (forall ((x (_ FloatingPoint 8 24)) (y (_ FloatingPoint 8 24)))
         (=> (and (fp.geq x ((_ to_fp 8 24) RNE 0.0))
                  (fp.leq x ((_ to_fp 8 24) RNE 1.0))
                  (fp.geq y ((_ to_fp 8 24) RNE 0.0))
                  (fp.leq y ((_ to_fp 8 24) RNE 1.0)))
             (inv x y))))

; Transition relation: x' = 0.75*x - 0.125*y, y' = x
(assert (forall ((x (_ FloatingPoint 8 24)) (y (_ FloatingPoint 8 24))
                 (x! (_ FloatingPoint 8 24)) (y! (_ FloatingPoint 8 24)))
         (=> (and (inv x y)
                  (fp.eq x! (fp.add RNE (fp.mul RNE ((_ to_fp 8 24) RNE 0.75) x)
                                         (fp.mul RNE ((_ to_fp 8 24) RNE -0.125) y)))
                  (fp.eq y! x))
             (inv x! y!))))

; Safety property: state variables should stay within [0.0, 1.0]
(assert (forall ((x (_ FloatingPoint 8 24)) (y (_ FloatingPoint 8 24)))
         (=> (inv x y)
             (and (fp.geq x ((_ to_fp 8 24) RNE 0.0))
                  (fp.leq x ((_ to_fp 8 24) RNE 1.0))
                  (fp.geq y ((_ to_fp 8 24) RNE 0.0))
                  (fp.leq y ((_ to_fp 8 24) RNE 1.0))))))

(check-sat)

