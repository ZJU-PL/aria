; ex3_leadlag.txt converted to CHC format
; Original: Linear discrete-time system with input
; State: x0, x1
; Input: in0 ∈ [-1.0, 1.0]
; Initial: x0 = 0, x1 = 0
; Transition: x0' = 0.499*x0 - 0.05*x1 + in0
;            x1' = 0.01*x0 + x1
; Safety: Verify boundedness of state variables

;(set-logic HORN)

(declare-fun inv ((_ FloatingPoint 8 24) (_ FloatingPoint 8 24) (_ FloatingPoint 8 24)) Bool)

; Initial state: x0 = 0, x1 = 0, in0 ∈ [-1.0, 1.0]
(assert (forall ((x0 (_ FloatingPoint 8 24)) (x1 (_ FloatingPoint 8 24)) (in0 (_ FloatingPoint 8 24)))
         (=> (and (fp.eq x0 ((_ to_fp 8 24) RNE 0.0))
                  (fp.eq x1 ((_ to_fp 8 24) RNE 0.0))
                  (fp.geq in0 ((_ to_fp 8 24) RNE (- 1.0)))
                  (fp.leq in0 ((_ to_fp 8 24) RNE 1.0)))
             (inv x0 x1 in0))))

; Transition relation: x0' = 0.499*x0 - 0.05*x1 + in0, x1' = 0.01*x0 + x1
; Input in0 can be any value in [-1.0, 1.0]
(assert (forall ((x0 (_ FloatingPoint 8 24)) (x1 (_ FloatingPoint 8 24)) (in0 (_ FloatingPoint 8 24))
                 (x0! (_ FloatingPoint 8 24)) (x1! (_ FloatingPoint 8 24)) (in0! (_ FloatingPoint 8 24)))
         (=> (and (inv x0 x1 in0)
                  (fp.eq x0! (fp.add RNE (fp.add RNE (fp.mul RNE ((_ to_fp 8 24) RNE 0.499) x0)
                                                     (fp.mul RNE ((_ to_fp 8 24) RNE -0.05) x1))
                                     in0!))
                  (fp.eq x1! (fp.add RNE (fp.mul RNE ((_ to_fp 8 24) RNE 0.01) x0) x1))
                  (fp.geq in0! ((_ to_fp 8 24) RNE (- 1.0)))
                  (fp.leq in0! ((_ to_fp 8 24) RNE 1.0)))
             (inv x0! x1! in0!))))

; Safety property: state variables should stay bounded
; For this example, we'll check if x0 and x1 stay within reasonable bounds
(assert (forall ((x0 (_ FloatingPoint 8 24)) (x1 (_ FloatingPoint 8 24)) (in0 (_ FloatingPoint 8 24)))
         (=> (inv x0 x1 in0)
             (and (fp.geq x0 ((_ to_fp 8 24) RNE -10.0))
                  (fp.leq x0 ((_ to_fp 8 24) RNE 10.0))
                  (fp.geq x1 ((_ to_fp 8 24) RNE -10.0))
                  (fp.leq x1 ((_ to_fp 8 24) RNE 10.0))))))

(check-sat)
