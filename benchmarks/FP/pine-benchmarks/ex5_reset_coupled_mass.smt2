; ex5_reset_coupled_mass.txt converted to CHC format
; Original: Conditional discrete-time system with reset
; State: x0, x1, x2, x3
; Input: in0 ∈ [-1.0, 1.0], in1 ∈ [-1.0, 1.0], r ∈ [0.0, 1.0]
; Initial: x0 = 0, x1 = 0, x2 = 0, x3 = 0, in0 ∈ [-1.0, 1.0], in1 ∈ [-1.0, 1.0], r ∈ [0.0, 1.0]
; Transition: if r > 0.5 then {x0' = 0.6227*x0 + 0.3871*x1 - 0.113*x2 + 0.0102*x3 + 0.3064*in0 + 0.1826*in1, x1' = -0.3407*x0 + 0.9103*x1 - 0.3388*x2 + 0.0649*x3 - 0.0054*in0 + 0.6731*in1, x2' = 0.0918*x0 - 0.0265*x1 - 0.7319*x2 + 0.2669*x3 + 0.0494*in0 + 1.6138*in1, x3' = 0.2643*x0 - 0.1298*x1 - 0.9903*x2 + 0.3331*x3 - 0.0531*in0 + 0.4012*in1}
;             else {x0' = 1, x1' = 1, x2' = 1, x3' = 1}
; Safety: Verify boundedness of state variables

(set-logic HORN)

(declare-fun inv ((_ FloatingPoint 8 24) (_ FloatingPoint 8 24) (_ FloatingPoint 8 24) (_ FloatingPoint 8 24) (_ FloatingPoint 8 24) (_ FloatingPoint 8 24) (_ FloatingPoint 8 24)) Bool)

; Initial state: x0 = 0, x1 = 0, x2 = 0, x3 = 0, in0 ∈ [-1.0, 1.0], in1 ∈ [-1.0, 1.0], r ∈ [0.0, 1.0]
(assert (forall ((x0 (_ FloatingPoint 8 24)) (x1 (_ FloatingPoint 8 24)) (x2 (_ FloatingPoint 8 24)) (x3 (_ FloatingPoint 8 24)) (in0 (_ FloatingPoint 8 24)) (in1 (_ FloatingPoint 8 24)) (r (_ FloatingPoint 8 24)))
         (=> (and (fp.eq x0 ((_ to_fp 8 24) RNE 0.0))
                  (fp.eq x1 ((_ to_fp 8 24) RNE 0.0))
                  (fp.eq x2 ((_ to_fp 8 24) RNE 0.0))
                  (fp.eq x3 ((_ to_fp 8 24) RNE 0.0))
                  (fp.geq in0 ((_ to_fp 8 24) RNE (- 1.0)))
                  (fp.leq in0 ((_ to_fp 8 24) RNE 1.0))
                  (fp.geq in1 ((_ to_fp 8 24) RNE (- 1.0)))
                  (fp.leq in1 ((_ to_fp 8 24) RNE 1.0))
                  (fp.geq r ((_ to_fp 8 24) RNE 0.0))
                  (fp.leq r ((_ to_fp 8 24) RNE 1.0)))
             (inv x0 x1 x2 x3 in0 in1 r))))

; Transition relation with conditional logic
(assert (forall ((x0 (_ FloatingPoint 8 24)) (x1 (_ FloatingPoint 8 24)) (x2 (_ FloatingPoint 8 24)) (x3 (_ FloatingPoint 8 24)) (in0 (_ FloatingPoint 8 24)) (in1 (_ FloatingPoint 8 24)) (r (_ FloatingPoint 8 24))
                 (x0! (_ FloatingPoint 8 24)) (x1! (_ FloatingPoint 8 24)) (x2! (_ FloatingPoint 8 24)) (x3! (_ FloatingPoint 8 24)) (in0! (_ FloatingPoint 8 24)) (in1! (_ FloatingPoint 8 24)) (r! (_ FloatingPoint 8 24)))
         (=> (and (inv x0 x1 x2 x3 in0 in1 r)
                  (fp.geq in0! ((_ to_fp 8 24) RNE (- 1.0)))
                  (fp.leq in0! ((_ to_fp 8 24) RNE 1.0))
                  (fp.geq in1! ((_ to_fp 8 24) RNE (- 1.0)))
                  (fp.leq in1! ((_ to_fp 8 24) RNE 1.0))
                  (fp.geq r! ((_ to_fp 8 24) RNE 0.0))
                  (fp.leq r! ((_ to_fp 8 24) RNE 1.0))
                  (ite (fp.gt r ((_ to_fp 8 24) RNE 0.5))
                       (and (fp.eq x0! (fp.add RNE (fp.add RNE (fp.add RNE (fp.add RNE (fp.add RNE (fp.mul RNE ((_ to_fp 8 24) RNE 0.6227) x0)
                                                                                      (fp.mul RNE ((_ to_fp 8 24) RNE 0.3871) x1))
                                                                         (fp.mul RNE ((_ to_fp 8 24) RNE -0.113) x2))
                                                            (fp.mul RNE ((_ to_fp 8 24) RNE 0.0102) x3))
                                                   (fp.mul RNE ((_ to_fp 8 24) RNE 0.3064) in0!))
                                          (fp.mul RNE ((_ to_fp 8 24) RNE 0.1826) in1!)))
                            (fp.eq x1! (fp.add RNE (fp.add RNE (fp.add RNE (fp.add RNE (fp.add RNE (fp.mul RNE ((_ to_fp 8 24) RNE -0.3407) x0)
                                                                                      (fp.mul RNE ((_ to_fp 8 24) RNE 0.9103) x1))
                                                                         (fp.mul RNE ((_ to_fp 8 24) RNE -0.3388) x2))
                                                            (fp.mul RNE ((_ to_fp 8 24) RNE 0.0649) x3))
                                                   (fp.mul RNE ((_ to_fp 8 24) RNE -0.0054) in0!))
                                          (fp.mul RNE ((_ to_fp 8 24) RNE 0.6731) in1!)))
                            (fp.eq x2! (fp.add RNE (fp.add RNE (fp.add RNE (fp.add RNE (fp.add RNE (fp.mul RNE ((_ to_fp 8 24) RNE 0.0918) x0)
                                                                                      (fp.mul RNE ((_ to_fp 8 24) RNE -0.0265) x1))
                                                                         (fp.mul RNE ((_ to_fp 8 24) RNE -0.7319) x2))
                                                            (fp.mul RNE ((_ to_fp 8 24) RNE 0.2669) x3))
                                                   (fp.mul RNE ((_ to_fp 8 24) RNE 0.0494) in0!))
                                          (fp.mul RNE ((_ to_fp 8 24) RNE 1.6138) in1!)))
                            (fp.eq x3! (fp.add RNE (fp.add RNE (fp.add RNE (fp.add RNE (fp.add RNE (fp.mul RNE ((_ to_fp 8 24) RNE 0.2643) x0)
                                                                                      (fp.mul RNE ((_ to_fp 8 24) RNE -0.1298) x1))
                                                                         (fp.mul RNE ((_ to_fp 8 24) RNE -0.9903) x2))
                                                            (fp.mul RNE ((_ to_fp 8 24) RNE 0.3331) x3))
                                                   (fp.mul RNE ((_ to_fp 8 24) RNE -0.0531) in0!))
                                          (fp.mul RNE ((_ to_fp 8 24) RNE 0.4012) in1!))))
                       (and (fp.eq x0! ((_ to_fp 8 24) RNE 1.0))
                            (fp.eq x1! ((_ to_fp 8 24) RNE 1.0))
                            (fp.eq x2! ((_ to_fp 8 24) RNE 1.0))
                            (fp.eq x3! ((_ to_fp 8 24) RNE 1.0)))))
             (inv x0! x1! x2! x3! in0! in1! r!))))

; Safety property: state variables should stay bounded
(assert (forall ((x0 (_ FloatingPoint 8 24)) (x1 (_ FloatingPoint 8 24)) (x2 (_ FloatingPoint 8 24)) (x3 (_ FloatingPoint 8 24)) (in0 (_ FloatingPoint 8 24)) (in1 (_ FloatingPoint 8 24)) (r (_ FloatingPoint 8 24)))
         (=> (inv x0 x1 x2 x3 in0 in1 r)
             (and (fp.geq x0 ((_ to_fp 8 24) RNE -10.0))
                  (fp.leq x0 ((_ to_fp 8 24) RNE 10.0))
                  (fp.geq x1 ((_ to_fp 8 24) RNE -10.0))
                  (fp.leq x1 ((_ to_fp 8 24) RNE 10.0))
                  (fp.geq x2 ((_ to_fp 8 24) RNE -10.0))
                  (fp.leq x2 ((_ to_fp 8 24) RNE 10.0))
                  (fp.geq x3 ((_ to_fp 8 24) RNE -10.0))
                  (fp.leq x3 ((_ to_fp 8 24) RNE 10.0))))))

(check-sat)

