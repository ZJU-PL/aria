(declare-const _a_0_ Real)
(declare-const _a_1_ Real)
(declare-const _a_2_ Real)
(declare-const _a_3_ Real)
(assert (forall ((x_0 Real) (n_0 Real) (y_0 Real) ) (=> (and (>= (+ (* 100. 1) (* -1 x_0) ) 0)(>= (+ (* 100. 1) (* -1 y_0) ) 0)) (and (>= (+ (* (+ -9. (* 2 _a_0_) (* 100. (*  (+ -1 (* 0.1 _a_1_))   (+ -1 (* 0.1 _a_1_))  ))) (*  n_0   n_0  )) (* (+ 90. _a_3_ (* -9. _a_1_) (* 2 _a_0_ (+ -10. _a_1_))) n_0) (* (+ _a_2_ (*  _a_0_   _a_0_  ) (* -9. _a_0_)) 1) (* 1 (*  n_0   n_0   n_0   n_0  )) (* (+ -20. (* 2 _a_1_)) (*  n_0   n_0   n_0  )) ) 0)))))
(check-sat)
(get-model)

