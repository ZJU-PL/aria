(set-logic BV)

(synth-inv inv_fun ((n (_ BitVec 64))(c (_ BitVec 64))))

(define-fun pre_fun ((n (_ BitVec 64)) (c (_ BitVec 64))) Bool
       ( and ( = c #x0000000000000000 ) ( bvugt n #x0000000000000000 ) ))
(define-fun trans_fun ((n! (_ BitVec 64)) (c! (_ BitVec 64)) (n (_ BitVec 64)) (c (_ BitVec 64))) Bool
       ( let ( ( a!1 ( or ( and ( bvugt c n ) ( = c! ( bvadd c #x0000000000000001 ) ) ( = n! n ) ) ( and ( = c n ) ( = c! #x0000000000000001 ) ( = n! n ) ) ) ) ) ( and a!1 ) ))
(define-fun post_fun ((n (_ BitVec 64)) (c (_ BitVec 64))) Bool
       ( and ( or ( = c n ) ( bvule c n ) ) ( not ( = c n ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

