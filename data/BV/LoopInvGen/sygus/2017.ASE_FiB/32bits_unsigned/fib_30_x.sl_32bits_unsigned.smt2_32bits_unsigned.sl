(set-logic BV)

(synth-inv inv_fun ((n (_ BitVec 32))(c (_ BitVec 32))(i (_ BitVec 32))))

(define-fun pre_fun ((n (_ BitVec 32)) (c (_ BitVec 32)) (i (_ BitVec 32))) Bool
       ( and ( = i #x00000000 ) ( = c #x00000000 ) ( bvugt n #x00000000 ) ))
(define-fun trans_fun ((n! (_ BitVec 32)) (c! (_ BitVec 32)) (i! (_ BitVec 32)) (n (_ BitVec 32)) (c (_ BitVec 32)) (i (_ BitVec 32))) Bool
       ( let ( ( a!1 ( or ( and ( bvuge i n ) ( = i! i ) ( = c! c ) ( = n! n ) ) ( and ( bvult i n ) ( = i! ( bvadd i #x00000001 ) ) ( = c! ( bvadd c i ) ) ( = n! n ) ) ) ) ) ( and a!1 ) ))
(define-fun post_fun ((n (_ BitVec 32)) (c (_ BitVec 32)) (i (_ BitVec 32))) Bool
       ( t r u e ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

