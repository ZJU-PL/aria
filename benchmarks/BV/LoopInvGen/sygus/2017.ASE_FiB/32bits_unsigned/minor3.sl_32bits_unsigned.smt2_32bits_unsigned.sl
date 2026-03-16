(set-logic BV)

(synth-inv inv_fun ((y (_ BitVec 32))(x (_ BitVec 32))))

(define-fun pre_fun ((y (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( and ( bvule x #x00000005 ) ( bvuge x #x00000004 ) ( bvule y #x00000005 ) ( bvuge y #x00000004 ) ))
(define-fun trans_fun ((y! (_ BitVec 32)) (x! (_ BitVec 32)) (y (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( let ( ( a!1 ( or ( and ( = x! ( bvsub x #x00000001 ) ) ( = y! y ) ( bvuge x #x00000000 ) ( bvuge y #x00000000 ) ) ( and ( = x! x ) ( = y! ( bvsub y #x00000001 ) ) ( bvult x #x00000000 ) ( bvuge y #x00000000 ) ) ( and ( = x! ( bvadd x #x00000001 ) ) ( = y! ( bvsub y #x00000001 ) ) ( bvult y #x00000000 ) ) ) ) ) ( and a!1 ) ))
(define-fun post_fun ((y (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( and ( = ( ( _ extract (_ bv31 32) (_ bv3 32) ) y ) #b00000000000000000000000000000 ) ( bvule ( ( _ extract (_ bv2 32) (_ bv0 32) ) y ) #b101 ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

