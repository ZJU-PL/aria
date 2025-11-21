; https://www.sfu.ca/math-coursenotes/Math%20157%20Course%20Notes/sec_Optimization.html -> 5.8.4
(declare-fun x () Real)
(assert (>= x 0))
(assert (<= (* x x) 1))
(maximize (+ (- (* 2 x x x)) (* 2 x)))
(check-sat)
(get-objectives)
(exit)
