; ex1_reset.txt converted to CHC format
; Original: Conditional discrete-time system with reset
; State: x, y
; Input: in0 ∈ [-1.0, 1.0], r ∈ [0.0, 1.0]
; Initial: x = 0, y = 0, in0 ∈ [-1.0, 1.0], r ∈ [0.0, 1.0]
; Transition: if r >= 0.5 then {x' = 1.5*x - 0.7*y + 1.6*in0, y' = x}
;             else {x' = 1, y' = 1}
; Safety: Verify boundedness of state variables

(set-logic HORN)

(declare-fun inv ((_ FloatingPoint 8 24) (_ FloatingPoint 8 24) (_ FloatingPoint 8 24) (_ FloatingPoint 8 24)) Bool)

; Initial state: x = 0, y = 0, in0 ∈ [-1.0, 1.0], r ∈ [0.0, 1.0]
(assert (forall ((x (_ FloatingPoint 8 24)) (y (_ FloatingPoint 8 24)) (in0 (_ FloatingPoint 8 24)) (r (_ FloatingPoint 8 24)))
         (=> (and (fp.eq x ((_ to_fp 8 24) RNE 0.0))
                  (fp.eq y ((_ to_fp 8 24) RNE 0.0))
                  (fp.geq in0 ((_ to_fp 8 24) RNE (- 1.0)))
                  (fp.leq in0 ((_ to_fp 8 24) RNE 1.0))
                  (fp.geq r ((_ to_fp 8 24) RNE 0.0))
                  (fp.leq r ((_ to_fp 8 24) RNE 1.0)))
             (inv x y in0 r))))

; Transition relation with conditional logic
; If r >= 0.5: x' = 1.5*x - 0.7*y + 1.6*in0, y' = x
; Else: x' = 1, y' = 1
; in0' and r' can be any value in their respective ranges
(assert (forall ((x (_ FloatingPoint 8 24)) (y (_ FloatingPoint 8 24)) (in0 (_ FloatingPoint 8 24)) (r (_ FloatingPoint 8 24))
                 (x! (_ FloatingPoint 8 24)) (y! (_ FloatingPoint 8 24)) (in0! (_ FloatingPoint 8 24)) (r! (_ FloatingPoint 8 24)))
         (=> (and (inv x y in0 r)
                  (fp.geq in0! ((_ to_fp 8 24) RNE (- 1.0)))
                  (fp.leq in0! ((_ to_fp 8 24) RNE 1.0))
                  (fp.geq r! ((_ to_fp 8 24) RNE 0.0))
                  (fp.leq r! ((_ to_fp 8 24) RNE 1.0))
                  (ite (fp.geq r ((_ to_fp 8 24) RNE 0.5))
                       (and (fp.eq x! (fp.add RNE (fp.add RNE (fp.mul RNE ((_ to_fp 8 24) RNE 1.5) x)
                                                              (fp.mul RNE ((_ to_fp 8 24) RNE -0.7) y))
                                                     (fp.mul RNE ((_ to_fp 8 24) RNE 1.6) in0!)))
                            (fp.eq y! x))
                       (and (fp.eq x! ((_ to_fp 8 24) RNE 1.0))
                            (fp.eq y! ((_ to_fp 8 24) RNE 1.0)))))
             (inv x! y! in0! r!))))

; Safety property: state variables should stay bounded
(assert (forall ((x (_ FloatingPoint 8 24)) (y (_ FloatingPoint 8 24)) (in0 (_ FloatingPoint 8 24)) (r (_ FloatingPoint 8 24)))
         (=> (inv x y in0 r)
             (and (fp.geq x ((_ to_fp 8 24) RNE -10.0))
                  (fp.leq x ((_ to_fp 8 24) RNE 10.0))
                  (fp.geq y ((_ to_fp 8 24) RNE -10.0))
                  (fp.leq y ((_ to_fp 8 24) RNE 10.0))))))

(check-sat)

