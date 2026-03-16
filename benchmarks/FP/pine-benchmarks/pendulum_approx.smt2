; pendulum_approx.txt converted to CHC format
; Original: Approximate nonlinear pendulum system
; State: u, v
; Initial: u = 0, v ∈ [2.0, 3.0]
; Transition: u' = u + 0.01*v
;            v' = v + 0.01*(-0.5*v - 9.81*(u - (u*u*u)/6.0 + (u*u*u*u*u)/120.0))
; Safety: Verify boundedness of state variables

(set-logic HORN)

(declare-fun inv ((_ FloatingPoint 8 24) (_ FloatingPoint 8 24)) Bool)

; Initial state: u = 0, v ∈ [2.0, 3.0]
(assert (forall ((u (_ FloatingPoint 8 24)) (v (_ FloatingPoint 8 24)))
         (=> (and (fp.eq u ((_ to_fp 8 24) RNE 0.0))
                  (fp.geq v ((_ to_fp 8 24) RNE 2.0))
                  (fp.leq v ((_ to_fp 8 24) RNE 3.0)))
             (inv u v))))

; Transition relation: u' = u + 0.01*v
; v' = v + 0.01*(-0.5*v - 9.81*(u - (u*u*u)/6.0 + (u*u*u*u*u)/120.0))
(assert (forall ((u (_ FloatingPoint 8 24)) (v (_ FloatingPoint 8 24))
                 (u! (_ FloatingPoint 8 24)) (v! (_ FloatingPoint 8 24)))
         (=> (and (inv u v)
                  (fp.eq u! (fp.add RNE u (fp.mul RNE ((_ to_fp 8 24) RNE 0.01) v)))
                  (fp.eq v! (fp.add RNE v (fp.mul RNE ((_ to_fp 8 24) RNE 0.01)
                                         (fp.sub RNE (fp.mul RNE ((_ to_fp 8 24) RNE -0.5) v)
                                                (fp.mul RNE ((_ to_fp 8 24) RNE 9.81)
                                                   (fp.add RNE (fp.sub RNE u (fp.div RNE (fp.mul RNE u (fp.mul RNE u u)) ((_ to_fp 8 24) RNE 6.0)))
                                                          (fp.div RNE (fp.mul RNE u (fp.mul RNE u (fp.mul RNE u (fp.mul RNE u u)))) ((_ to_fp 8 24) RNE 120.0))))))))
             (inv u! v!))))

; Safety property: state variables should stay bounded
(assert (forall ((u (_ FloatingPoint 8 24)) (v (_ FloatingPoint 8 24)))
         (=> (inv u v)
             (and (fp.geq u ((_ to_fp 8 24) RNE -10.0))
                  (fp.leq u ((_ to_fp 8 24) RNE 10.0))
                  (fp.geq v ((_ to_fp 8 24) RNE -10.0))
                  (fp.leq v ((_ to_fp 8 24) RNE 10.0))))))

(check-sat)

