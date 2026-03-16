; filter_mine2_nondet.txt converted to CHC format
; Original: Filter system with nondeterministic input
; State: s0, s1
; Input: n ∈ [-0.1, 0.1]
; Initial: s0, s1 ∈ [-0.1, 0.1]
; Transition: s1' = s0
;            s0' = 1.5*s0 - 0.7*s1 + n
; Safety: Verify state variables stay within [-0.1, 0.1]

(set-logic HORN)

(declare-fun inv ((_ FloatingPoint 8 24) (_ FloatingPoint 8 24) (_ FloatingPoint 8 24)) Bool)

; Initial state: s0, s1 ∈ [-0.1, 0.1], n ∈ [-0.1, 0.1]
(assert (forall ((s0 (_ FloatingPoint 8 24)) (s1 (_ FloatingPoint 8 24)) (n (_ FloatingPoint 8 24)))
         (=> (and (fp.geq s0 ((_ to_fp 8 24) RNE (- 0.1)))
                  (fp.leq s0 ((_ to_fp 8 24) RNE 0.1))
                  (fp.geq s1 ((_ to_fp 8 24) RNE (- 0.1)))
                  (fp.leq s1 ((_ to_fp 8 24) RNE 0.1))
                  (fp.geq n ((_ to_fp 8 24) RNE (- 0.1)))
                  (fp.leq n ((_ to_fp 8 24) RNE 0.1)))
             (inv s0 s1 n))))

; Transition relation: s1' = s0, s0' = 1.5*s0 - 0.7*s1 + n
; Input n can be any value in [-0.1, 0.1]
(assert (forall ((s0 (_ FloatingPoint 8 24)) (s1 (_ FloatingPoint 8 24)) (n (_ FloatingPoint 8 24))
                 (s0! (_ FloatingPoint 8 24)) (s1! (_ FloatingPoint 8 24)) (n! (_ FloatingPoint 8 24)))
         (=> (and (inv s0 s1 n)
                  (fp.eq s1! s0)
                  (fp.eq s0! (fp.add RNE (fp.add RNE (fp.mul RNE ((_ to_fp 8 24) RNE 1.5) s0)
                                                     (fp.mul RNE ((_ to_fp 8 24) RNE -0.7) s1))
                                    n!))
                  (fp.geq n! ((_ to_fp 8 24) RNE (- 0.1)))
                  (fp.leq n! ((_ to_fp 8 24) RNE 0.1)))
             (inv s0! s1! n!))))

; Safety property: state variables should stay within [-0.1, 0.1]
(assert (forall ((s0 (_ FloatingPoint 8 24)) (s1 (_ FloatingPoint 8 24)) (n (_ FloatingPoint 8 24)))
         (=> (inv s0 s1 n)
             (and (fp.geq s0 ((_ to_fp 8 24) RNE (- 0.1)))
                  (fp.leq s0 ((_ to_fp 8 24) RNE 0.1))
                  (fp.geq s1 ((_ to_fp 8 24) RNE (- 0.1)))
                  (fp.leq s1 ((_ to_fp 8 24) RNE 0.1))))))

(check-sat)

