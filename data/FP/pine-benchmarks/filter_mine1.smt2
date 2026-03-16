; filter_mine1.txt converted to CHC format
; Original: Filter system
; State: x, y
; Initial: x, y ∈ [-1.0, 1.0]
; Transition: x' = 0.68*(x - y)
;            y' = 0.68*(x + y)
; Safety: Verify state variables stay within [-1.0, 1.0]

(set-logic HORN)

(declare-fun inv ((_ FloatingPoint 8 24) (_ FloatingPoint 8 24)) Bool)

; Initial state: x, y ∈ [-1.0, 1.0]
(assert (forall ((x (_ FloatingPoint 8 24)) (y (_ FloatingPoint 8 24)))
         (=> (and (fp.geq x ((_ to_fp 8 24) RNE (- 1.0)))
                  (fp.leq x ((_ to_fp 8 24) RNE 1.0))
                  (fp.geq y ((_ to_fp 8 24) RNE (- 1.0)))
                  (fp.leq y ((_ to_fp 8 24) RNE 1.0)))
             (inv x y))))

; Transition relation: x' = 0.68*(x - y), y' = 0.68*(x + y)
(assert (forall ((x (_ FloatingPoint 8 24)) (y (_ FloatingPoint 8 24))
                 (x! (_ FloatingPoint 8 24)) (y! (_ FloatingPoint 8 24)))
         (=> (and (inv x y)
                  (fp.eq x! (fp.mul RNE ((_ to_fp 8 24) RNE 0.68) (fp.sub RNE x y)))
                  (fp.eq y! (fp.mul RNE ((_ to_fp 8 24) RNE 0.68) (fp.add RNE x y))))
             (inv x! y!))))

; Safety property: state variables should stay within [-1.0, 1.0]
(assert (forall ((x (_ FloatingPoint 8 24)) (y (_ FloatingPoint 8 24)))
         (=> (inv x y)
             (and (fp.geq x ((_ to_fp 8 24) RNE (- 1.0)))
                  (fp.leq x ((_ to_fp 8 24) RNE 1.0))
                  (fp.geq y ((_ to_fp 8 24) RNE (- 1.0)))
                  (fp.leq y ((_ to_fp 8 24) RNE 1.0))))))

(check-sat)

