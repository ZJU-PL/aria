(declare-const _a_0_ Real)
(assert (forall ((x_0 Real) (n_0 Real) ) (=> (and (>= (+ (* 100. 1) (* -1 x_0) ) 0)) (and (>= (+ (* 1 (*  n_0   n_0  )) (* -5. n_0) (* _a_0_ 1) ) 0)))))
(check-sat)
(get-model)

