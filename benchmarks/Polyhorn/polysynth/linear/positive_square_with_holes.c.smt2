(declare-const _a_0_ Real)
(declare-const _a_1_ Real)
(assert (forall ((n_0 Real) (x_0 Real) ) (=> (and (>= (+ (* 100. 1) (* -1 x_0) ) 0)) (and (>= (+ (* 1 (*  n_0   n_0  )) (* (+ -5. _a_0_) n_0) (* _a_1_ 1) ) 0)))))
(check-sat)
(get-model)

