(set-logic BV)

(synth-inv inv_fun ((n (_ BitVec 64))(c (_ BitVec 64))(i (_ BitVec 64))))

(define-fun pre_fun ((n (_ BitVec 64)) (c (_ BitVec 64)) (i (_ BitVec 64))) Bool
       ( and ( = i #x0000000000000000 ) ( = c #x0000000000000000 ) ( bvugt n #x0000000000000000 ) ))
(define-fun trans_fun ((n! (_ BitVec 64)) (c! (_ BitVec 64)) (i! (_ BitVec 64)) (n (_ BitVec 64)) (c (_ BitVec 64)) (i (_ BitVec 64))) Bool
       ( let ( ( a!1 ( or ( and ( bvuge i n ) ( = i! i ) ( = c! c ) ( = n! n ) ) ( and ( bvult i n ) ( = i! ( bvadd i #x0000000000000001 ) ) ( = c! ( bvadd c i ) ) ( = n! n ) ) ) ) ) ( and a!1 ) ))
(define-fun post_fun ((n (_ BitVec 64)) (c (_ BitVec 64)) (i (_ BitVec 64))) Bool
       ( t r u e ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

