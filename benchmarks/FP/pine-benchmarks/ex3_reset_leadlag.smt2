; ex3_reset_leadlag.txt converted to CHC format
; Original: Conditional discrete-time system with reset
; State: x0, x1
; Input: in0 ∈ [-1.0, 1.0], r ∈ [0.0, 1.0]
; Initial: x0 = 0, x1 = 0, in0 ∈ [-1.0, 1.0], r ∈ [0.0, 1.0]
; Transition: if r > 0.5 then {x0' = 0.499*x0 - 0.05*x1 + in0, x1' = 0.01*x0 + x1}
;             else {x0' = 1, x1' = 1}
; Safety: Verify boundedness of state variables

(set-logic HORN)

(declare-fun inv ((_ FloatingPoint 8 24) (_ FloatingPoint 8 24) (_ FloatingPoint 8 24) (_ FloatingPoint 8 24)) Bool)

; Initial state: x0 = 0, x1 = 0, in0 ∈ [-1.0, 1.0], r ∈ [0.0, 1.0]
(assert (forall ((x0 (_ FloatingPoint 8 24)) (x1 (_ FloatingPoint 8 24)) (in0 (_ FloatingPoint 8 24)) (r (_ FloatingPoint 8 24)))
         (=> (and (fp.eq x0 ((_ to_fp 8 24) RNE 0.0))
                  (fp.eq x1 ((_ to_fp 8 24) RNE 0.0))
                  (fp.geq in0 ((_ to_fp 8 24) RNE (- 1.0)))
                  (fp.leq in0 ((_ to_fp 8 24) RNE 1.0))
                  (fp.geq r ((_ to_fp 8 24) RNE 0.0))
                  (fp.leq r ((_ to_fp 8 24) RNE 1.0)))
             (inv x0 x1 in0 r))))

; Transition relation with conditional logic
(assert (forall ((x0 (_ FloatingPoint 8 24)) (x1 (_ FloatingPoint 8 24)) (in0 (_ FloatingPoint 8 24)) (r (_ FloatingPoint 8 24))
                 (x0! (_ FloatingPoint 8 24)) (x1! (_ FloatingPoint 8 24)) (in0! (_ FloatingPoint 8 24)) (r! (_ FloatingPoint 8 24)))
         (=> (and (inv x0 x1 in0 r)
                  (fp.geq in0! ((_ to_fp 8 24) RNE (- 1.0)))
                  (fp.leq in0! ((_ to_fp 8 24) RNE 1.0))
                  (fp.geq r! ((_ to_fp 8 24) RNE 0.0))
                  (fp.leq r! ((_ to_fp 8 24) RNE 1.0))
                  (ite (fp.gt r ((_ to_fp 8 24) RNE 0.5))
                       (and (fp.eq x0! (fp.add RNE (fp.add RNE (fp.mul RNE ((_ to_fp 8 24) RNE 0.499) x0)
                                                              (fp.mul RNE ((_ to_fp 8 24) RNE -0.05) x1))
                                                     in0!))
                            (fp.eq x1! (fp.add RNE (fp.mul RNE ((_ to_fp 8 24) RNE 0.01) x0) x1)))
                       (and (fp.eq x0! ((_ to_fp 8 24) RNE 1.0))
                            (fp.eq x1! ((_ to_fp 8 24) RNE 1.0)))))
             (inv x0! x1! in0! r!))))

; Safety property: state variables should stay bounded
(assert (forall ((x0 (_ FloatingPoint 8 24)) (x1 (_ FloatingPoint 8 24)) (in0 (_ FloatingPoint 8 24)) (r (_ FloatingPoint 8 24)))
         (=> (inv x0 x1 in0 r)
             (and (fp.geq x0 ((_ to_fp 8 24) RNE -10.0))
                  (fp.leq x0 ((_ to_fp 8 24) RNE 10.0))
                  (fp.geq x1 ((_ to_fp 8 24) RNE -10.0))
                  (fp.leq x1 ((_ to_fp 8 24) RNE 10.0))))))

(check-sat)

