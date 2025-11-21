; boxed OMT(BV) problem
(set-logic QF_BV)
(set-info :smt-lib-version 2.6)
(set-info :category "industrial")

(declare-fun k!1 () (_ BitVec 8))
(declare-fun k!2 () (_ BitVec 8))
(declare-fun k!3 () (_ BitVec 8))
(declare-fun k!4 () (_ BitVec 8))
(declare-fun k!5 () (_ BitVec 8))
(declare-fun k!6 () (_ BitVec 8))
(declare-fun k!7 () (_ BitVec 8))

(assert (and (or false
         (and (bvuge k!1 #xfe) (bvule k!1 #xff))
         (and (bvuge k!2 #x80) (bvule k!2 #xff))
         (and (bvuge k!3 #x81) (bvule k!3 #xff))
         (and (bvuge k!4 #x81) (bvule k!4 #xff))
         (and (bvuge k!5 #x81) (bvule k!5 #xff)))))
(assert (not (bvuge k!6 #x81)))
(assert (not (bvuge k!7 #x81)))

(maximize k!1)
(maximize k!2)
(maximize k!3)
(maximize k!4)
(maximize k!5)
(maximize k!6)
(maximize k!7)

(check-sat)
(get-model)