(set-logic BV)

(synth-inv inv_fun ((size (_ BitVec 32))(sn (_ BitVec 32))(i (_ BitVec 32))))

(define-fun pre_fun ((size (_ BitVec 32)) (sn (_ BitVec 32)) (i (_ BitVec 32))) Bool
       ( and ( = sn #x00000000 ) ( = i #x00000001 ) ))
(define-fun trans_fun ((size! (_ BitVec 32)) (sn! (_ BitVec 32)) (i! (_ BitVec 32)) (size (_ BitVec 32)) (sn (_ BitVec 32)) (i (_ BitVec 32))) Bool
       ( and ( = size! size ) ( = i! ( bvadd i #x00000001 ) ) ( bvsle i size ) ( = sn! ( bvadd sn #x00000001 ) ) ))
(define-fun post_fun ((size (_ BitVec 32)) (sn (_ BitVec 32)) (i (_ BitVec 32))) Bool
       ( or ( = sn size ) ( = sn #x00000000 ) ( bvsle i size ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

