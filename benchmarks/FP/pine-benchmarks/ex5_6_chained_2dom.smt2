; ex5_6_chained_2dom.txt converted to CHC format
; Original: Chained system with two domains
; State: x5_0, x5_1, x5_2, x5_3, x6_0, x6_1, x6_2, x6_3, x6_4
; Input: in5_0 ∈ [-1.0, 1.0], in5_1 ∈ [-1.0, 1.0], in6_0 ∈ [-1.0, 1.0] (where in6_0 = 0.3*x5_1)
; Initial: All state variables = 0
; Transition: x5_0' = 0.6227*x5_0 + 0.3871*x5_1 - 0.113*x5_2 + 0.0102*x5_3 + 0.3064*in5_0 + 0.1826*in5_1
;            x5_1' = -0.3407*x5_0 + 0.9103*x5_1 - 0.3388*x5_2 + 0.0649*x5_3 - 0.0054*in5_0 + 0.6731*in5_1
;            x5_2' = 0.0918*x5_0 - 0.0265*x5_1 - 0.7319*x5_2 + 0.2669*x5_3 + 0.0494*in5_0 + 1.6138*in5_1
;            x5_3' = 0.2643*x5_0 - 0.1298*x5_1 - 0.9903*x5_2 + 0.3331*x5_3 - 0.0531*in5_0 + 0.4012*in5_1
;            in6_0 = 0.3*x5_1
;            x6_0' = 0.4250*x6_0 + 0.8131*in6_0
;            x6_1' = 0.3167*x6_0 + 0.1016*x6_1 - 0.4444*x6_2 + 0.1807*in6_0
;            x6_2' = 0.1278*x6_0 + 0.4444*x6_1 + 0.8207*x6_2 + 0.0729*in6_0
;            x6_3' = 0.0365*x6_0 + 0.1270*x6_1 + 0.5202*x6_2 + 0.4163*x6_3 - 0.5714*x6_4 + 0.0208*in6_0
;            x6_4' = 0.0147*x6_0 + 0.0512*x6_1 + 0.2099*x6_2 + 0.57104*x6_3 + 0.7694*x6_4 + 0.0084*in6_0
; Safety: Verify boundedness of state variables

(set-logic HORN)

(declare-fun inv ((_ FloatingPoint 8 24) (_ FloatingPoint 8 24) (_ FloatingPoint 8 24) (_ FloatingPoint 8 24) (_ FloatingPoint 8 24) (_ FloatingPoint 8 24) (_ FloatingPoint 8 24) (_ FloatingPoint 8 24) (_ FloatingPoint 8 24) (_ FloatingPoint 8 24) (_ FloatingPoint 8 24)) Bool)

; Initial state: All state variables = 0, in5_0 ∈ [-1.0, 1.0], in5_1 ∈ [-1.0, 1.0]
(assert (forall ((x5_0 (_ FloatingPoint 8 24)) (x5_1 (_ FloatingPoint 8 24)) (x5_2 (_ FloatingPoint 8 24)) (x5_3 (_ FloatingPoint 8 24))
                 (x6_0 (_ FloatingPoint 8 24)) (x6_1 (_ FloatingPoint 8 24)) (x6_2 (_ FloatingPoint 8 24)) (x6_3 (_ FloatingPoint 8 24)) (x6_4 (_ FloatingPoint 8 24))
                 (in5_0 (_ FloatingPoint 8 24)) (in5_1 (_ FloatingPoint 8 24)))
         (=> (and (fp.eq x5_0 ((_ to_fp 8 24) RNE 0.0))
                  (fp.eq x5_1 ((_ to_fp 8 24) RNE 0.0))
                  (fp.eq x5_2 ((_ to_fp 8 24) RNE 0.0))
                  (fp.eq x5_3 ((_ to_fp 8 24) RNE 0.0))
                  (fp.eq x6_0 ((_ to_fp 8 24) RNE 0.0))
                  (fp.eq x6_1 ((_ to_fp 8 24) RNE 0.0))
                  (fp.eq x6_2 ((_ to_fp 8 24) RNE 0.0))
                  (fp.eq x6_3 ((_ to_fp 8 24) RNE 0.0))
                  (fp.eq x6_4 ((_ to_fp 8 24) RNE 0.0))
                  (fp.geq in5_0 ((_ to_fp 8 24) RNE (- 1.0)))
                  (fp.leq in5_0 ((_ to_fp 8 24) RNE 1.0))
                  (fp.geq in5_1 ((_ to_fp 8 24) RNE (- 1.0)))
                  (fp.leq in5_1 ((_ to_fp 8 24) RNE 1.0)))
             (inv x5_0 x5_1 x5_2 x5_3 x6_0 x6_1 x6_2 x6_3 x6_4 in5_0 in5_1))))

; Transition relation
; in6_0 = 0.3*x5_1 (computed from current state)
(assert (forall ((x5_0 (_ FloatingPoint 8 24)) (x5_1 (_ FloatingPoint 8 24)) (x5_2 (_ FloatingPoint 8 24)) (x5_3 (_ FloatingPoint 8 24))
                 (x6_0 (_ FloatingPoint 8 24)) (x6_1 (_ FloatingPoint 8 24)) (x6_2 (_ FloatingPoint 8 24)) (x6_3 (_ FloatingPoint 8 24)) (x6_4 (_ FloatingPoint 8 24))
                 (in5_0 (_ FloatingPoint 8 24)) (in5_1 (_ FloatingPoint 8 24))
                 (x5_0! (_ FloatingPoint 8 24)) (x5_1! (_ FloatingPoint 8 24)) (x5_2! (_ FloatingPoint 8 24)) (x5_3! (_ FloatingPoint 8 24))
                 (x6_0! (_ FloatingPoint 8 24)) (x6_1! (_ FloatingPoint 8 24)) (x6_2! (_ FloatingPoint 8 24)) (x6_3! (_ FloatingPoint 8 24)) (x6_4! (_ FloatingPoint 8 24))
                 (in5_0! (_ FloatingPoint 8 24)) (in5_1! (_ FloatingPoint 8 24)))
         (=> (and (inv x5_0 x5_1 x5_2 x5_3 x6_0 x6_1 x6_2 x6_3 x6_4 in5_0 in5_1)
                  (fp.geq in5_0! ((_ to_fp 8 24) RNE (- 1.0)))
                  (fp.leq in5_0! ((_ to_fp 8 24) RNE 1.0))
                  (fp.geq in5_1! ((_ to_fp 8 24) RNE (- 1.0)))
                  (fp.leq in5_1! ((_ to_fp 8 24) RNE 1.0))
                  ; x5 transitions
                  (fp.eq x5_0! (fp.add RNE (fp.add RNE (fp.add RNE (fp.add RNE (fp.add RNE (fp.mul RNE ((_ to_fp 8 24) RNE 0.6227) x5_0)
                                                                                      (fp.mul RNE ((_ to_fp 8 24) RNE 0.3871) x5_1))
                                                                         (fp.mul RNE ((_ to_fp 8 24) RNE -0.113) x5_2))
                                                            (fp.mul RNE ((_ to_fp 8 24) RNE 0.0102) x5_3))
                                                   (fp.mul RNE ((_ to_fp 8 24) RNE 0.3064) in5_0!))
                                          (fp.mul RNE ((_ to_fp 8 24) RNE 0.1826) in5_1!)))
                  (fp.eq x5_1! (fp.add RNE (fp.add RNE (fp.add RNE (fp.add RNE (fp.add RNE (fp.mul RNE ((_ to_fp 8 24) RNE -0.3407) x5_0)
                                                                                      (fp.mul RNE ((_ to_fp 8 24) RNE 0.9103) x5_1))
                                                                         (fp.mul RNE ((_ to_fp 8 24) RNE -0.3388) x5_2))
                                                            (fp.mul RNE ((_ to_fp 8 24) RNE 0.0649) x5_3))
                                                   (fp.mul RNE ((_ to_fp 8 24) RNE -0.0054) in5_0!))
                                          (fp.mul RNE ((_ to_fp 8 24) RNE 0.6731) in5_1!)))
                  (fp.eq x5_2! (fp.add RNE (fp.add RNE (fp.add RNE (fp.add RNE (fp.add RNE (fp.mul RNE ((_ to_fp 8 24) RNE 0.0918) x5_0)
                                                                                      (fp.mul RNE ((_ to_fp 8 24) RNE -0.0265) x5_1))
                                                                         (fp.mul RNE ((_ to_fp 8 24) RNE -0.7319) x5_2))
                                                            (fp.mul RNE ((_ to_fp 8 24) RNE 0.2669) x5_3))
                                                   (fp.mul RNE ((_ to_fp 8 24) RNE 0.0494) in5_0!))
                                          (fp.mul RNE ((_ to_fp 8 24) RNE 1.6138) in5_1!)))
                  (fp.eq x5_3! (fp.add RNE (fp.add RNE (fp.add RNE (fp.add RNE (fp.add RNE (fp.mul RNE ((_ to_fp 8 24) RNE 0.2643) x5_0)
                                                                                      (fp.mul RNE ((_ to_fp 8 24) RNE -0.1298) x5_1))
                                                                         (fp.mul RNE ((_ to_fp 8 24) RNE -0.9903) x5_2))
                                                            (fp.mul RNE ((_ to_fp 8 24) RNE 0.3331) x5_3))
                                                   (fp.mul RNE ((_ to_fp 8 24) RNE -0.0531) in5_0!))
                                          (fp.mul RNE ((_ to_fp 8 24) RNE 0.4012) in5_1!)))
                  ; in6_0 computed from x5_1!
                  ; x6 transitions (using in6_0 = 0.3*x5_1!)
                  (fp.eq x6_0! (fp.add RNE (fp.mul RNE ((_ to_fp 8 24) RNE 0.4250) x6_0)
                                         (fp.mul RNE ((_ to_fp 8 24) RNE 0.8131) (fp.mul RNE ((_ to_fp 8 24) RNE 0.3) x5_1!))))
                  (fp.eq x6_1! (fp.add RNE (fp.add RNE (fp.add RNE (fp.mul RNE ((_ to_fp 8 24) RNE 0.3167) x6_0)
                                                                 (fp.mul RNE ((_ to_fp 8 24) RNE 0.1016) x6_1))
                                                      (fp.mul RNE ((_ to_fp 8 24) RNE -0.4444) x6_2))
                                         (fp.mul RNE ((_ to_fp 8 24) RNE 0.1807) (fp.mul RNE ((_ to_fp 8 24) RNE 0.3) x5_1!))))
                  (fp.eq x6_2! (fp.add RNE (fp.add RNE (fp.add RNE (fp.mul RNE ((_ to_fp 8 24) RNE 0.1278) x6_0)
                                                                 (fp.mul RNE ((_ to_fp 8 24) RNE 0.4444) x6_1))
                                                      (fp.mul RNE ((_ to_fp 8 24) RNE 0.8207) x6_2))
                                         (fp.mul RNE ((_ to_fp 8 24) RNE 0.0729) (fp.mul RNE ((_ to_fp 8 24) RNE 0.3) x5_1!))))
                  (fp.eq x6_3! (fp.add RNE (fp.add RNE (fp.add RNE (fp.add RNE (fp.add RNE (fp.mul RNE ((_ to_fp 8 24) RNE 0.0365) x6_0)
                                                                                      (fp.mul RNE ((_ to_fp 8 24) RNE 0.1270) x6_1))
                                                                         (fp.mul RNE ((_ to_fp 8 24) RNE 0.5202) x6_2))
                                                            (fp.mul RNE ((_ to_fp 8 24) RNE 0.4163) x6_3))
                                                   (fp.mul RNE ((_ to_fp 8 24) RNE -0.5714) x6_4))
                                          (fp.mul RNE ((_ to_fp 8 24) RNE 0.0208) (fp.mul RNE ((_ to_fp 8 24) RNE 0.3) x5_1!))))
                  (fp.eq x6_4! (fp.add RNE (fp.add RNE (fp.add RNE (fp.add RNE (fp.add RNE (fp.mul RNE ((_ to_fp 8 24) RNE 0.0147) x6_0)
                                                                                      (fp.mul RNE ((_ to_fp 8 24) RNE 0.0512) x6_1))
                                                                         (fp.mul RNE ((_ to_fp 8 24) RNE 0.2099) x6_2))
                                                            (fp.mul RNE ((_ to_fp 8 24) RNE 0.57104) x6_3))
                                                   (fp.mul RNE ((_ to_fp 8 24) RNE 0.7694) x6_4))
                                          (fp.mul RNE ((_ to_fp 8 24) RNE 0.0084) (fp.mul RNE ((_ to_fp 8 24) RNE 0.3) x5_1!)))))
             (inv x5_0! x5_1! x5_2! x5_3! x6_0! x6_1! x6_2! x6_3! x6_4! in5_0! in5_1!))))

; Safety property: state variables should stay bounded
(assert (forall ((x5_0 (_ FloatingPoint 8 24)) (x5_1 (_ FloatingPoint 8 24)) (x5_2 (_ FloatingPoint 8 24)) (x5_3 (_ FloatingPoint 8 24))
                 (x6_0 (_ FloatingPoint 8 24)) (x6_1 (_ FloatingPoint 8 24)) (x6_2 (_ FloatingPoint 8 24)) (x6_3 (_ FloatingPoint 8 24)) (x6_4 (_ FloatingPoint 8 24))
                 (in5_0 (_ FloatingPoint 8 24)) (in5_1 (_ FloatingPoint 8 24)))
         (=> (inv x5_0 x5_1 x5_2 x5_3 x6_0 x6_1 x6_2 x6_3 x6_4 in5_0 in5_1)
             (and (fp.geq x5_0 ((_ to_fp 8 24) RNE -10.0))
                  (fp.leq x5_0 ((_ to_fp 8 24) RNE 10.0))
                  (fp.geq x5_1 ((_ to_fp 8 24) RNE -10.0))
                  (fp.leq x5_1 ((_ to_fp 8 24) RNE 10.0))
                  (fp.geq x5_2 ((_ to_fp 8 24) RNE -10.0))
                  (fp.leq x5_2 ((_ to_fp 8 24) RNE 10.0))
                  (fp.geq x5_3 ((_ to_fp 8 24) RNE -10.0))
                  (fp.leq x5_3 ((_ to_fp 8 24) RNE 10.0))
                  (fp.geq x6_0 ((_ to_fp 8 24) RNE -10.0))
                  (fp.leq x6_0 ((_ to_fp 8 24) RNE 10.0))
                  (fp.geq x6_1 ((_ to_fp 8 24) RNE -10.0))
                  (fp.leq x6_1 ((_ to_fp 8 24) RNE 10.0))
                  (fp.geq x6_2 ((_ to_fp 8 24) RNE -10.0))
                  (fp.leq x6_2 ((_ to_fp 8 24) RNE 10.0))
                  (fp.geq x6_3 ((_ to_fp 8 24) RNE -10.0))
                  (fp.leq x6_3 ((_ to_fp 8 24) RNE 10.0))
                  (fp.geq x6_4 ((_ to_fp 8 24) RNE -10.0))
                  (fp.leq x6_4 ((_ to_fp 8 24) RNE 10.0))))))

(check-sat)

