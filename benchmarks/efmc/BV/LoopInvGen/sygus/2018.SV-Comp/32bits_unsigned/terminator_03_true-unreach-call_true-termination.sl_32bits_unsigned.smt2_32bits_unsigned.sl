(set-logic BV)

(synth-inv inv_fun ((y (_ BitVec 32))(x (_ BitVec 32))))

(define-fun pre_fun ((y (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( bvule y #x000f4240 ))
(define-fun trans_fun ((y! (_ BitVec 32)) (x! (_ BitVec 32)) (y (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( and ( bvult x #x00000064 ) ( bvugt y #x00000000 ) ( = x! ( bvadd x y ) ) ( = y! y ) ))
(define-fun post_fun ((y (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( let ( ( a!1 ( or ( = y #x00000000 ) ( and ( not ( = y #x00000000 ) ) ( bvule #x00000064 x ) ) ) ) ) ( or ( not a!1 ) ( = y #x00000000 ) ( and ( not ( = y #x00000000 ) ) ( bvule #x00000064 x ) ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

