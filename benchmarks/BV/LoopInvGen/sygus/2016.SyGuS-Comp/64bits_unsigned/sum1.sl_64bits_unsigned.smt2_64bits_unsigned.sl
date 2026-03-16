(set-logic BV)

(synth-inv inv_fun ((sn (_ BitVec 64))(n (_ BitVec 64))(i (_ BitVec 64))))

(define-fun pre_fun ((sn (_ BitVec 64)) (n (_ BitVec 64)) (i (_ BitVec 64))) Bool
       ( and ( = sn #x0000000000000000 ) ( = i #x0000000000000001 ) ))
(define-fun trans_fun ((sn! (_ BitVec 64)) (n! (_ BitVec 64)) (i! (_ BitVec 64)) (sn (_ BitVec 64)) (n (_ BitVec 64)) (i (_ BitVec 64))) Bool
       ( and ( = n! n ) ( = i! ( bvadd i #x0000000000000001 ) ) ( bvule i n ) ( = sn! ( bvadd sn #x0000000000000001 ) ) ))
(define-fun post_fun ((sn (_ BitVec 64)) (n (_ BitVec 64)) (i (_ BitVec 64))) Bool
       ( or ( = sn #x0000000000000000 ) ( bvule i n ) ( = sn n ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

