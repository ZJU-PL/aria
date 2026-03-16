(set-logic BV)

(synth-inv inv_fun ((sn (_ BitVec 64))(x (_ BitVec 64))))

(define-fun pre_fun ((sn (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( and ( = sn #x0000000000000000 ) ( = x #x0000000000000000 ) ))
(define-fun trans_fun ((sn! (_ BitVec 64)) (x! (_ BitVec 64)) (sn (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( and ( = x! ( bvadd x #x0000000000000001 ) ) ( = sn! ( bvadd sn #x0000000000000001 ) ) ))
(define-fun post_fun ((sn (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( or ( = sn x ) ( = sn #xffffffffffffffff ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

