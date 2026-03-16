(set-logic BV)

(synth-inv inv_fun ((c (_ BitVec 32))))

(define-fun pre_fun ((c (_ BitVec 32))) Bool
       ( = c #x00000000 ))
(define-fun trans_fun ((c! (_ BitVec 32)) (c (_ BitVec 32))) Bool
       ( let ( ( a!1 ( or ( and ( not ( = c #x00000028 ) ) ( = c! ( bvadd c #x00000001 ) ) ) ( and ( = c #x00000028 ) ( = c! #x00000001 ) ) ) ) ) ( and a!1 ) ))
(define-fun post_fun ((c (_ BitVec 32))) Bool
       ( let ( ( a!1 ( and ( not ( = c #x00000028 ) ) ( or ( not ( bvsle #x00000000 c ) ) ( not ( bvsle c #x00000028 ) ) ) ) ) ) ( not a!1 ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

