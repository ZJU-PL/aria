(set-logic BV)

(synth-inv inv_fun ((v3 (_ BitVec 64))(v2 (_ BitVec 64))(v1 (_ BitVec 64))(n (_ BitVec 64))(c (_ BitVec 64))))

(define-fun pre_fun ((v3 (_ BitVec 64)) (v2 (_ BitVec 64)) (v1 (_ BitVec 64)) (n (_ BitVec 64)) (c (_ BitVec 64))) Bool
       ( and ( = c #x0000000000000000 ) ( bvsgt n #x0000000000000000 ) ))
(define-fun trans_fun ((v3! (_ BitVec 64)) (v2! (_ BitVec 64)) (v1! (_ BitVec 64)) (n! (_ BitVec 64)) (c! (_ BitVec 64)) (v3 (_ BitVec 64)) (v2 (_ BitVec 64)) (v1 (_ BitVec 64)) (n (_ BitVec 64)) (c (_ BitVec 64))) Bool
       ( let ( ( a!1 ( or ( and ( not ( = c n ) ) ( = c! ( bvadd c #x0000000000000001 ) ) ) ( and ( = c n ) ( = c! #x0000000000000001 ) ) ) ) ) ( and a!1 ) ))
(define-fun post_fun ((v3 (_ BitVec 64)) (v2 (_ BitVec 64)) (v1 (_ BitVec 64)) (n (_ BitVec 64)) (c (_ BitVec 64))) Bool
       ( let ( ( a!1 ( and ( not ( = c n ) ) ( or ( not ( bvsle #x0000000000000000 c ) ) ( not ( bvsle c n ) ) ) ) ) ) ( let ( ( a!2 ( or a!1 ( and ( = c n ) ( not ( bvsle n #xffffffffffffffff ) ) ) ) ) ) ( not a!2 ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

