; ex2_reset.txt converted to CHC format
; Original: Conditional discrete-time system with reset
; State: x0, x1, x2, x3
; Input: in0 ∈ [-1.0, 1.0], r ∈ [0.0, 1.0]
; Initial: x0 = 0, x1 = 0, x2 = 0, x3 = 0, in0 ∈ [-1.0, 1.0], r ∈ [0.0, 1.0]
; Transition: if r > 0.5 then {x0' = 1.5*x0 - 0.7*x1 - 0.7*x2 + 0.4*x3 + 0.5*in0, x1' = x0, x2' = in0, x3' = x2}
;             else {x0' = 1, x1' = 1, x2' = 1, x3' = 1}
; Safety: Verify boundedness of state variables

(set-logic HORN)

(declare-fun inv ((_ FloatingPoint 8 24) (_ FloatingPoint 8 24) (_ FloatingPoint 8 24) (_ FloatingPoint 8 24) (_ FloatingPoint 8 24) (_ FloatingPoint 8 24)) Bool)

; Initial state: x0 = 0, x1 = 0, x2 = 0, x3 = 0, in0 ∈ [-1.0, 1.0], r ∈ [0.0, 1.0]
(assert (forall ((x0 (_ FloatingPoint 8 24)) (x1 (_ FloatingPoint 8 24)) (x2 (_ FloatingPoint 8 24)) (x3 (_ FloatingPoint 8 24)) (in0 (_ FloatingPoint 8 24)) (r (_ FloatingPoint 8 24)))
         (=> (and (fp.eq x0 ((_ to_fp 8 24) RNE 0.0))
                  (fp.eq x1 ((_ to_fp 8 24) RNE 0.0))
                  (fp.eq x2 ((_ to_fp 8 24) RNE 0.0))
                  (fp.eq x3 ((_ to_fp 8 24) RNE 0.0))
                  (fp.geq in0 ((_ to_fp 8 24) RNE (- 1.0)))
                  (fp.leq in0 ((_ to_fp 8 24) RNE 1.0))
                  (fp.geq r ((_ to_fp 8 24) RNE 0.0))
                  (fp.leq r ((_ to_fp 8 24) RNE 1.0)))
             (inv x0 x1 x2 x3 in0 r))))

; Transition relation with conditional logic
(assert (forall ((x0 (_ FloatingPoint 8 24)) (x1 (_ FloatingPoint 8 24)) (x2 (_ FloatingPoint 8 24)) (x3 (_ FloatingPoint 8 24)) (in0 (_ FloatingPoint 8 24)) (r (_ FloatingPoint 8 24))
                 (x0! (_ FloatingPoint 8 24)) (x1! (_ FloatingPoint 8 24)) (x2! (_ FloatingPoint 8 24)) (x3! (_ FloatingPoint 8 24)) (in0! (_ FloatingPoint 8 24)) (r! (_ FloatingPoint 8 24)))
         (=> (and (inv x0 x1 x2 x3 in0 r)
                  (fp.geq in0! ((_ to_fp 8 24) RNE (- 1.0)))
                  (fp.leq in0! ((_ to_fp 8 24) RNE 1.0))
                  (fp.geq r! ((_ to_fp 8 24) RNE 0.0))
                  (fp.leq r! ((_ to_fp 8 24) RNE 1.0))
                  (ite (fp.gt r ((_ to_fp 8 24) RNE 0.5))
                       (and (fp.eq x0! (fp.add RNE (fp.add RNE (fp.add RNE (fp.add RNE (fp.mul RNE ((_ to_fp 8 24) RNE 1.5) x0)
                                                                                    (fp.mul RNE ((_ to_fp 8 24) RNE -0.7) x1))
                                                                       (fp.mul RNE ((_ to_fp 8 24) RNE -0.7) x2))
                                                          (fp.mul RNE ((_ to_fp 8 24) RNE 0.4) x3))
                                                 (fp.mul RNE ((_ to_fp 8 24) RNE 0.5) in0!)))
                            (fp.eq x1! x0)
                            (fp.eq x2! in0!)
                            (fp.eq x3! x2))
                       (and (fp.eq x0! ((_ to_fp 8 24) RNE 1.0))
                            (fp.eq x1! ((_ to_fp 8 24) RNE 1.0))
                            (fp.eq x2! ((_ to_fp 8 24) RNE 1.0))
                            (fp.eq x3! ((_ to_fp 8 24) RNE 1.0)))))
             (inv x0! x1! x2! x3! in0! r!))))

; Safety property: state variables should stay bounded
(assert (forall ((x0 (_ FloatingPoint 8 24)) (x1 (_ FloatingPoint 8 24)) (x2 (_ FloatingPoint 8 24)) (x3 (_ FloatingPoint 8 24)) (in0 (_ FloatingPoint 8 24)) (r (_ FloatingPoint 8 24)))
         (=> (inv x0 x1 x2 x3 in0 r)
             (and (fp.geq x0 ((_ to_fp 8 24) RNE -10.0))
                  (fp.leq x0 ((_ to_fp 8 24) RNE 10.0))
                  (fp.geq x1 ((_ to_fp 8 24) RNE -10.0))
                  (fp.leq x1 ((_ to_fp 8 24) RNE 10.0))
                  (fp.geq x2 ((_ to_fp 8 24) RNE -10.0))
                  (fp.leq x2 ((_ to_fp 8 24) RNE 10.0))
                  (fp.geq x3 ((_ to_fp 8 24) RNE -10.0))
                  (fp.leq x3 ((_ to_fp 8 24) RNE 10.0))))))

(check-sat)

