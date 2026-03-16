(set-logic BV)

(synth-inv inv_fun ((sn (_ BitVec 32))(i (_ BitVec 32))))

(define-fun pre_fun ((sn (_ BitVec 32)) (i (_ BitVec 32))) Bool
       ( and ( = sn #x00000000 ) ( = i #x00000001 ) ))
(define-fun trans_fun ((sn! (_ BitVec 32)) (i! (_ BitVec 32)) (sn (_ BitVec 32)) (i (_ BitVec 32))) Bool
       ( and ( bvsle i #x00000008 ) ( ite ( bvslt i #x00000004 ) ( = sn! ( bvadd sn #x00000002 ) ) ( = sn! sn ) ) ( = i! ( bvadd i #x00000001 ) ) ))
(define-fun post_fun ((sn (_ BitVec 32)) (i (_ BitVec 32))) Bool
       ( or ( bvsle i #x00000008 ) ( = sn #x00000000 ) ( = sn #x00000010 ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

