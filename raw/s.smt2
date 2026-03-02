(set-option :print-success false)
(set-logic QF_LIA)
(declare-fun x1 () Int)
(declare-fun x2 () Int)
(assert (let (
    (x3 
        (> (ite (= x1 x2) x1 x2) 0))
    (x5
        1
    )
    ) x3))
(check-sat)
(exit)