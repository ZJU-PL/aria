; filter_mine2.txt converted to CHC format
; Original: Filter system
; State: s0, s1
; Initial: s0, s1 ∈ [-0.1, 0.1]
; Transition: s0' = 1.5*s0 - 0.7*s1
;            s1' = s0
; Safety: Verify state variables stay within [-0.1, 0.1]

(set-logic HORN)

(declare-fun inv ((_ FloatingPoint 8 24) (_ FloatingPoint 8 24)) Bool)

; Initial state: s0, s1 ∈ [-0.1, 0.1]
(assert (forall ((s0 (_ FloatingPoint 8 24)) (s1 (_ FloatingPoint 8 24)))
         (=> (and (fp.geq s0 ((_ to_fp 8 24) RNE (- 0.1)))
                  (fp.leq s0 ((_ to_fp 8 24) RNE 0.1))
                  (fp.geq s1 ((_ to_fp 8 24) RNE (- 0.1)))
                  (fp.leq s1 ((_ to_fp 8 24) RNE 0.1)))
             (inv s0 s1))))

; Transition relation: s0' = 1.5*s0 - 0.7*s1, s1' = s0
(assert (forall ((s0 (_ FloatingPoint 8 24)) (s1 (_ FloatingPoint 8 24))
                 (s0! (_ FloatingPoint 8 24)) (s1! (_ FloatingPoint 8 24)))
         (=> (and (inv s0 s1)
                  (fp.eq s0! (fp.add RNE (fp.mul RNE ((_ to_fp 8 24) RNE 1.5) s0)
                                         (fp.mul RNE ((_ to_fp 8 24) RNE -0.7) s1)))
                  (fp.eq s1! s0))
             (inv s0! s1!))))

; Safety property: state variables should stay within [-0.1, 0.1]
(assert (forall ((s0 (_ FloatingPoint 8 24)) (s1 (_ FloatingPoint 8 24)))
         (=> (inv s0 s1)
             (and (fp.geq s0 ((_ to_fp 8 24) RNE (- 0.1)))
                  (fp.leq s0 ((_ to_fp 8 24) RNE 0.1))
                  (fp.geq s1 ((_ to_fp 8 24) RNE (- 0.1)))
                  (fp.leq s1 ((_ to_fp 8 24) RNE 0.1))))))

(check-sat)

