(set-logic BV)

(synth-inv inv_fun ((n (_ BitVec 64))(y (_ BitVec 64))(x (_ BitVec 64))(i (_ BitVec 64))))

(define-fun pre_fun ((n (_ BitVec 64)) (y (_ BitVec 64)) (x (_ BitVec 64)) (i (_ BitVec 64))) Bool
       ( and ( = i #x0000000000000000 ) ( = x #x0000000000000000 ) ( = y #x0000000000000000 ) ( bvugt n #x0000000000000000 ) ))
(define-fun trans_fun ((n! (_ BitVec 64)) (y! (_ BitVec 64)) (x! (_ BitVec 64)) (i! (_ BitVec 64)) (n (_ BitVec 64)) (y (_ BitVec 64)) (x (_ BitVec 64)) (i (_ BitVec 64))) Bool
       ( and ( = i! ( bvadd i #x0000000000000001 ) ) ( = x! x ) ( = y! y ) ( = n! n ) ))
(define-fun post_fun ((n (_ BitVec 64)) (y (_ BitVec 64)) (x (_ BitVec 64)) (i (_ BitVec 64))) Bool
       ( = x #x0000000000000000 ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

