(set-logic BV)

(synth-inv inv_fun ((v3 (_ BitVec 32))(v2 (_ BitVec 32))(v1 (_ BitVec 32))(n (_ BitVec 32))(c (_ BitVec 32))))

(define-fun pre_fun ((v3 (_ BitVec 32)) (v2 (_ BitVec 32)) (v1 (_ BitVec 32)) (n (_ BitVec 32)) (c (_ BitVec 32))) Bool
       ( and ( = c #x00000000 ) ( bvsgt n #x00000000 ) ))
(define-fun trans_fun ((v3! (_ BitVec 32)) (v2! (_ BitVec 32)) (v1! (_ BitVec 32)) (n! (_ BitVec 32)) (c! (_ BitVec 32)) (v3 (_ BitVec 32)) (v2 (_ BitVec 32)) (v1 (_ BitVec 32)) (n (_ BitVec 32)) (c (_ BitVec 32))) Bool
       ( let ( ( a!1 ( or ( and ( not ( = c n ) ) ( = c! ( bvadd c #x00000001 ) ) ) ( and ( = c n ) ( = c! #x00000001 ) ) ) ) ) ( and a!1 ) ))
(define-fun post_fun ((v3 (_ BitVec 32)) (v2 (_ BitVec 32)) (v1 (_ BitVec 32)) (n (_ BitVec 32)) (c (_ BitVec 32))) Bool
       ( let ( ( a!1 ( and ( not ( = c n ) ) ( or ( not ( bvsle #x00000000 c ) ) ( not ( bvsle c n ) ) ) ) ) ) ( let ( ( a!2 ( or a!1 ( and ( = c n ) ( not ( bvsle n #xffffffff ) ) ) ) ) ) ( not a!2 ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

