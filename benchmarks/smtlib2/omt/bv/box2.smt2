; boxed OMT(BV) problem
(set-logic QF_BV)
(set-info :smt-lib-version 2.6)
(set-info :category "industrial")
(set-option :opt.priority box)

(declare-fun k!1 () (_ BitVec 8))
(declare-fun k!2 () (_ BitVec 8))
(declare-fun k!3 () (_ BitVec 8))
(declare-fun k!4 () (_ BitVec 8))

; Hard upper bounds that prevent all from being 255
(assert (bvule k!1 #xf8))
(assert (bvule k!2 #xec))
(assert (bvule k!3 #xe0))
; k!2 and k!4 conflict: if k!2 is at its max, k!4 must be lower
(assert (or (bvult k!2 #xec) (bvult k!4 #xc0)))
; k!1 and k!3 conflict: if k!1 is at its max, k!3 must be lower
(assert (or (bvult k!1 #xf8) (bvult k!3 #xd0)))
; Lower bounds to ensure non-trivial solutions
(assert (bvuge k!1 #xa0))
(assert (bvuge k!2 #x90))
(assert (bvuge k!3 #x80))
(assert (bvuge k!4 #x70))

(maximize k!1)
(maximize k!2)
(maximize k!3)
(maximize k!4)

(check-sat)
(get-model)
