(set-logic BV)

(synth-inv inv_fun ((sn (_ BitVec 32))(x (_ BitVec 32))))

(define-fun pre_fun ((sn (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( and ( = sn #x00000000 ) ( = x #x00000000 ) ))
(define-fun trans_fun ((sn! (_ BitVec 32)) (x! (_ BitVec 32)) (sn (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( and ( = x! ( bvadd x #x00000001 ) ) ( = sn! ( bvadd sn #x00000001 ) ) ))
(define-fun post_fun ((sn (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( or ( = sn #xffffffff ) ( = sn x ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

