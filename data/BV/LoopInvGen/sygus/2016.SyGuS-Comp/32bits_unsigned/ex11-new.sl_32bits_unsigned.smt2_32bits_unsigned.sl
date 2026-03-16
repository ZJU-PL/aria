(set-logic BV)

(synth-inv inv_fun ((c (_ BitVec 32))))

(define-fun pre_fun ((c (_ BitVec 32))) Bool
       ( = c #x00000000 ))
(define-fun trans_fun ((c! (_ BitVec 32)) (c (_ BitVec 32))) Bool
       ( let ( ( a!1 ( or ( and ( not ( = c #x00000028 ) ) ( = c! ( bvadd c #x00000001 ) ) ) ( and ( = c #x00000028 ) ( = c! #x00000001 ) ) ) ) ) ( and a!1 ) ))
(define-fun post_fun ((c (_ BitVec 32))) Bool
       ( let ( ( a!1 ( not ( and ( = ( ( _ extract (_ bv31 32) (_ bv6 32) ) c ) #b00000000000000000000000000 ) ( bvule ( ( _ extract (_ bv5 32) (_ bv0 32) ) c ) #b101000 ) ) ) ) ) ( not ( and ( not ( = c #x00000028 ) ) a!1 ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

