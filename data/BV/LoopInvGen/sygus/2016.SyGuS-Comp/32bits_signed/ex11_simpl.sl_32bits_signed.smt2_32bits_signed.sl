(set-logic BV)

(synth-inv inv_fun ((n (_ BitVec 32))(c (_ BitVec 32))))

(define-fun pre_fun ((n (_ BitVec 32)) (c (_ BitVec 32))) Bool
       ( and ( = c #x00000000 ) ( bvsgt n #x00000000 ) ))
(define-fun trans_fun ((n! (_ BitVec 32)) (c! (_ BitVec 32)) (n (_ BitVec 32)) (c (_ BitVec 32))) Bool
       ( let ( ( a!1 ( or ( and ( not ( = c n ) ) ( = c! ( bvadd c #x00000001 ) ) ( = n! n ) ) ( and ( = c n ) ( = c! #x00000001 ) ( = n! n ) ) ) ) ) ( and a!1 ) ))
(define-fun post_fun ((n (_ BitVec 32)) (c (_ BitVec 32))) Bool
       ( and ( or ( = c n ) ( and ( bvsle #x00000000 c ) ( bvsle c n ) ) ) ( or ( not ( = c n ) ) ( not ( bvsle n #xffffffff ) ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

