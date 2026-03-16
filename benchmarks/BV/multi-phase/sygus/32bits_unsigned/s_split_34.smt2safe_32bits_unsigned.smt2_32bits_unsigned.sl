(set-logic BV)

(synth-inv inv_fun ((y (_ BitVec 32))(x (_ BitVec 32))))

(define-fun pre_fun ((y (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( and ( = x #x00000000 ) ( = y #x00000001 ) ))
(define-fun trans_fun ((y! (_ BitVec 32)) (x! (_ BitVec 32)) (y (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( let ( ( a!1 ( and ( bvugt x! ( bvadd #x00000001 ( bvnot #x00000064 ) ) ) ( bvult x! #x00000064 ) ) ) ) ( let ( ( a!2 ( = y! ( ite a!1 y ( bvadd #x00000001 ( bvnot y ) ) ) ) ) ) ( and ( = x! ( bvadd x y ) ) a!2 ) ) ))
(define-fun post_fun ((y (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( and ( bvule #xffffff9c x ) ( = ( ( _ extract (_ bv31 32) (_ bv7 32) ) x ) #b0000000000000000000000000 ) ( bvule ( ( _ extract (_ bv6 32) (_ bv0 32) ) x ) #b1100100 ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

