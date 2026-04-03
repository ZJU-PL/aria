(set-logic BV)

(synth-inv inv_fun ((v3 (_ BitVec 32))(v2 (_ BitVec 32))(v1 (_ BitVec 32))(size (_ BitVec 32))(sn (_ BitVec 32))(i (_ BitVec 32))))

(define-fun pre_fun ((v3 (_ BitVec 32)) (v2 (_ BitVec 32)) (v1 (_ BitVec 32)) (size (_ BitVec 32)) (sn (_ BitVec 32)) (i (_ BitVec 32))) Bool
       ( and ( = sn #x00000000 ) ( = i #x00000001 ) ))
(define-fun trans_fun ((v3! (_ BitVec 32)) (v2! (_ BitVec 32)) (v1! (_ BitVec 32)) (size! (_ BitVec 32)) (sn! (_ BitVec 32)) (i! (_ BitVec 32)) (v3 (_ BitVec 32)) (v2 (_ BitVec 32)) (v1 (_ BitVec 32)) (size (_ BitVec 32)) (sn (_ BitVec 32)) (i (_ BitVec 32))) Bool
       ( and ( = size! size ) ( = i! ( bvadd i #x00000001 ) ) ( bvsle i size ) ( = sn! ( bvadd sn #x00000001 ) ) ))
(define-fun post_fun ((v3 (_ BitVec 32)) (v2 (_ BitVec 32)) (v1 (_ BitVec 32)) (size (_ BitVec 32)) (sn (_ BitVec 32)) (i (_ BitVec 32))) Bool
       ( or ( = sn #x00000000 ) ( bvsle i size ) ( = sn size ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

