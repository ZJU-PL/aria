(set-logic BV)

(synth-inv inv_fun ((c (_ BitVec 64))))

(define-fun pre_fun ((c (_ BitVec 64))) Bool
       ( = c #x0000000000000000 ))
(define-fun trans_fun ((c! (_ BitVec 64)) (c (_ BitVec 64))) Bool
       ( let ( ( a!1 ( or ( and ( not ( = c #x0000000000000028 ) ) ( = c! ( bvadd c #x0000000000000001 ) ) ) ( and ( = c #x0000000000000028 ) ( = c! #x0000000000000001 ) ) ) ) ) ( and a!1 ) ))
(define-fun post_fun ((c (_ BitVec 64))) Bool
       ( let ( ( a!1 ( not ( and ( = ( ( _ extract (_ bv63 64) (_ bv6 64) ) c ) #b0000000000000000000000000000000000000000000000000000000000 ) ( bvule ( ( _ extract (_ bv5 64) (_ bv0 64) ) c ) #b101000 ) ) ) ) ) ( not ( and ( not ( = c #x0000000000000028 ) ) a!1 ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

