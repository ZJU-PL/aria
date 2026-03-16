(set-logic BV)

(synth-inv inv_fun ((v3 (_ BitVec 32))(v2 (_ BitVec 32))(v1 (_ BitVec 32))(sn (_ BitVec 32))(x (_ BitVec 32))))

(define-fun pre_fun ((v3 (_ BitVec 32)) (v2 (_ BitVec 32)) (v1 (_ BitVec 32)) (sn (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( and ( = sn #x00000000 ) ( = x #x00000000 ) ))
(define-fun trans_fun ((v3! (_ BitVec 32)) (v2! (_ BitVec 32)) (v1! (_ BitVec 32)) (sn! (_ BitVec 32)) (x! (_ BitVec 32)) (v3 (_ BitVec 32)) (v2 (_ BitVec 32)) (v1 (_ BitVec 32)) (sn (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( and ( = x! ( bvadd x #x00000001 ) ) ( = sn! ( bvadd sn #x00000001 ) ) ))
(define-fun post_fun ((v3 (_ BitVec 32)) (v2 (_ BitVec 32)) (v1 (_ BitVec 32)) (sn (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( or ( = sn x ) ( = sn #xffffffff ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

