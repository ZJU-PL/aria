(declare-const _a_0_ Real)
(declare-const _a_1_ Real)
(declare-const _a_2_ Real)
(declare-const _a_3_ Real)
(assert (forall ((x_0 Real) ) (=> (and (>= (* 1. 1)  0)(>= (* -1 x_0)  0)) (and (>= (+ (* (+ 5. (* -1 _a_1_)) 1) (* (* -1 _a_0_) x_0) (* (* -1 _a_2_) (*  x_0   x_0  )) (* (+ 1 (* -1 _a_3_)) (*  x_0   x_0   x_0  )) ) 0)))))
(assert (forall ((x_0 Real) ) (=> (and (>= (* 1. 1)  0)(>= (* 1 x_0)  0)) (and (>= (+ (* (+ 5. (* -1 _a_1_)) 1) (* (+ 1 (* -1 _a_0_)) x_0) (* (* -1 _a_2_) (*  x_0   x_0  )) (* (+ 1 (* -1 _a_3_)) (*  x_0   x_0   x_0  )) ) 0)))))
(check-sat)
(get-model)

