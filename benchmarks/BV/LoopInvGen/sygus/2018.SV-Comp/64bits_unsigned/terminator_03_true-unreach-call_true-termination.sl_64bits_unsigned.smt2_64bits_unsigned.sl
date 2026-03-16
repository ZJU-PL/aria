(set-logic BV)

(synth-inv inv_fun ((y (_ BitVec 64))(x (_ BitVec 64))))

(define-fun pre_fun ((y (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( bvule y #x00000000000f4240 ))
(define-fun trans_fun ((y! (_ BitVec 64)) (x! (_ BitVec 64)) (y (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( and ( bvult x #x0000000000000064 ) ( bvugt y #x0000000000000000 ) ( = x! ( bvadd x y ) ) ( = y! y ) ))
(define-fun post_fun ((y (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( let ( ( a!1 ( or ( = y #x0000000000000000 ) ( and ( not ( = y #x0000000000000000 ) ) ( bvule #x0000000000000064 x ) ) ) ) ) ( or ( not a!1 ) ( = y #x0000000000000000 ) ( and ( not ( = y #x0000000000000000 ) ) ( bvule #x0000000000000064 x ) ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

