; boxed OMT(BV) problem
(set-logic QF_BV)
(set-info :smt-lib-version 2.6)
(set-info :category "industrial")
(set-option :opt.priority box)

(declare-fun k!1 () (_ BitVec 8))
(declare-fun k!2 () (_ BitVec 8))
(declare-fun k!3 () (_ BitVec 8))
(declare-fun k!4 () (_ BitVec 8))
(declare-fun k!5 () (_ BitVec 8))
(declare-fun k!6 () (_ BitVec 8))

; Hard upper bounds on some variables
(assert (bvule k!1 #xf5))
(assert (bvule k!3 #xe8))
(assert (bvule k!5 #xd0))
; k!1 and k!2 conflict: if k!1 is at max, k!2 must be lower
(assert (or (bvult k!1 #xf5) (bvult k!2 #xea)))
; k!2 and k!4 conflict: cannot both be high
(assert (or (bvult k!2 #xf0) (bvult k!4 #xc0)))
; k!3 and k!6 conflict: if k!3 is at max, k!6 must be lower
(assert (or (bvult k!3 #xe8) (bvult k!6 #xb0)))
; k!4 and k!5 conflict: if k!4 is high, k!5 must be lower
(assert (or (bvult k!4 #xf0) (bvult k!5 #xcf)))
; Lower bounds
(assert (bvuge k!1 #xa0))
(assert (bvuge k!2 #x90))
(assert (bvuge k!3 #x80))
(assert (bvuge k!4 #x70))
(assert (bvuge k!5 #x60))
(assert (bvuge k!6 #x50))

(maximize k!1)
(maximize k!2)
(maximize k!3)
(maximize k!4)
(maximize k!5)
(maximize k!6)

(check-sat)
(get-model)
