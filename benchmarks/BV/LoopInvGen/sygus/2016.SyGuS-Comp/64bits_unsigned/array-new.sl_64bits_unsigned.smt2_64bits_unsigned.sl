(set-logic BV)

(synth-inv inv_fun ((z (_ BitVec 64))(y (_ BitVec 64))(x (_ BitVec 64))))

(define-fun pre_fun ((z (_ BitVec 64)) (y (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( = x #x0000000000000000 ))
(define-fun trans_fun ((z! (_ BitVec 64)) (y! (_ BitVec 64)) (x! (_ BitVec 64)) (z (_ BitVec 64)) (y (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( let ( ( a!1 ( or ( and ( = x! ( bvadd x #x0000000000000001 ) ) ( = y! z! ) ( bvule z! y ) ( bvult x #x00000000000001f4 ) ) ( and ( = x! ( bvadd x #x0000000000000001 ) ) ( = y! y ) ( bvugt z! y ) ( bvult x #x00000000000001f4 ) ) ) ) ) ( and a!1 ) ))
(define-fun post_fun ((z (_ BitVec 64)) (y (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( not ( and ( bvule #x00000000000001f4 x ) ( not ( bvule y z ) ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

