; ex2.txt converted to CHC format
; Original: Linear discrete-time system
; State: x0, x1, x2, x3
; Input: in0 ∈ [-1.0, 1.0]
; Initial: x0 = 0, x1 = 0, x2 = 0, x3 = 0
; Transition: x0' = 1.5*x0 - 0.7*x1 - 0.7*x2 + 0.4*x3 + 0.5*in0
;            x1' = x0
;            x2' = in0
;            x3' = x2
; Safety: Verify boundedness of state variables

(set-logic HORN)

(declare-fun inv ((_ FloatingPoint 8 24) (_ FloatingPoint 8 24) (_ FloatingPoint 8 24) (_ FloatingPoint 8 24) (_ FloatingPoint 8 24)) Bool)

; Initial state: x0 = 0, x1 = 0, x2 = 0, x3 = 0, in0 ∈ [-1.0, 1.0]
(assert (forall ((x0 (_ FloatingPoint 8 24)) (x1 (_ FloatingPoint 8 24)) (x2 (_ FloatingPoint 8 24)) (x3 (_ FloatingPoint 8 24)) (in0 (_ FloatingPoint 8 24)))
         (=> (and (fp.eq x0 ((_ to_fp 8 24) RNE 0.0))
                  (fp.eq x1 ((_ to_fp 8 24) RNE 0.0))
                  (fp.eq x2 ((_ to_fp 8 24) RNE 0.0))
                  (fp.eq x3 ((_ to_fp 8 24) RNE 0.0))
                  (fp.geq in0 ((_ to_fp 8 24) RNE (- 1.0)))
                  (fp.leq in0 ((_ to_fp 8 24) RNE 1.0)))
             (inv x0 x1 x2 x3 in0))))

; Transition relation
(assert (forall ((x0 (_ FloatingPoint 8 24)) (x1 (_ FloatingPoint 8 24)) (x2 (_ FloatingPoint 8 24)) (x3 (_ FloatingPoint 8 24)) (in0 (_ FloatingPoint 8 24))
                 (x0! (_ FloatingPoint 8 24)) (x1! (_ FloatingPoint 8 24)) (x2! (_ FloatingPoint 8 24)) (x3! (_ FloatingPoint 8 24)) (in0! (_ FloatingPoint 8 24)))
         (=> (and (inv x0 x1 x2 x3 in0)
                  (fp.eq x0! (fp.add RNE (fp.add RNE (fp.add RNE (fp.add RNE (fp.mul RNE ((_ to_fp 8 24) RNE 1.5) x0)
                                                                           (fp.mul RNE ((_ to_fp 8 24) RNE -0.7) x1))
                                                      (fp.mul RNE ((_ to_fp 8 24) RNE -0.7) x2))
                                         (fp.mul RNE ((_ to_fp 8 24) RNE 0.4) x3))
                            (fp.mul RNE ((_ to_fp 8 24) RNE 0.5) in0!)))
                  (fp.eq x1! x0)
                  (fp.eq x2! in0!)
                  (fp.eq x3! x2)
                  (fp.geq in0! ((_ to_fp 8 24) RNE (- 1.0)))
                  (fp.leq in0! ((_ to_fp 8 24) RNE 1.0)))
             (inv x0! x1! x2! x3! in0!))))

; Safety property: state variables should stay bounded
(assert (forall ((x0 (_ FloatingPoint 8 24)) (x1 (_ FloatingPoint 8 24)) (x2 (_ FloatingPoint 8 24)) (x3 (_ FloatingPoint 8 24)) (in0 (_ FloatingPoint 8 24)))
         (=> (inv x0 x1 x2 x3 in0)
             (and (fp.geq x0 ((_ to_fp 8 24) RNE -10.0))
                  (fp.leq x0 ((_ to_fp 8 24) RNE 10.0))
                  (fp.geq x1 ((_ to_fp 8 24) RNE -10.0))
                  (fp.leq x1 ((_ to_fp 8 24) RNE 10.0))
                  (fp.geq x2 ((_ to_fp 8 24) RNE -10.0))
                  (fp.leq x2 ((_ to_fp 8 24) RNE 10.0))
                  (fp.geq x3 ((_ to_fp 8 24) RNE -10.0))
                  (fp.leq x3 ((_ to_fp 8 24) RNE 10.0))))))

(check-sat)

