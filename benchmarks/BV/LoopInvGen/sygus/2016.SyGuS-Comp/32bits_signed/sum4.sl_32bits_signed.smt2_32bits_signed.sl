(set-logic BV)

(synth-inv inv_fun ((sn (_ BitVec 32))(i (_ BitVec 32))))

(define-fun pre_fun ((sn (_ BitVec 32)) (i (_ BitVec 32))) Bool
       ( and ( = sn #x00000000 ) ( = i #x00000001 ) ))
(define-fun trans_fun ((sn! (_ BitVec 32)) (i! (_ BitVec 32)) (sn (_ BitVec 32)) (i (_ BitVec 32))) Bool
       ( and ( = i! ( bvadd i #x00000001 ) ) ( bvsle i #x00000008 ) ( = sn! ( bvadd sn #x00000001 ) ) ))
(define-fun post_fun ((sn (_ BitVec 32)) (i (_ BitVec 32))) Bool
       ( or ( bvsle i #x00000008 ) ( = sn #x00000008 ) ( = sn #x00000000 ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

