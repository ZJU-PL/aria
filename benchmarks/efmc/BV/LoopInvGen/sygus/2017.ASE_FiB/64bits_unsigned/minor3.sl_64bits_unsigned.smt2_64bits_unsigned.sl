(set-logic BV)

(synth-inv inv_fun ((y (_ BitVec 64))(x (_ BitVec 64))))

(define-fun pre_fun ((y (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( and ( bvule x #x0000000000000005 ) ( bvuge x #x0000000000000004 ) ( bvule y #x0000000000000005 ) ( bvuge y #x0000000000000004 ) ))
(define-fun trans_fun ((y! (_ BitVec 64)) (x! (_ BitVec 64)) (y (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( let ( ( a!1 ( or ( and ( = x! ( bvsub x #x0000000000000001 ) ) ( = y! y ) ( bvuge x #x0000000000000000 ) ( bvuge y #x0000000000000000 ) ) ( and ( = x! x ) ( = y! ( bvsub y #x0000000000000001 ) ) ( bvult x #x0000000000000000 ) ( bvuge y #x0000000000000000 ) ) ( and ( = x! ( bvadd x #x0000000000000001 ) ) ( = y! ( bvsub y #x0000000000000001 ) ) ( bvult y #x0000000000000000 ) ) ) ) ) ( and a!1 ) ))
(define-fun post_fun ((y (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( and ( = ( ( _ extract (_ bv63 64) (_ bv3 64) ) y ) #b0000000000000000000000000000000000000000000000000000000000000 ) ( bvule ( ( _ extract (_ bv2 64) (_ bv0 64) ) y ) #b101 ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

