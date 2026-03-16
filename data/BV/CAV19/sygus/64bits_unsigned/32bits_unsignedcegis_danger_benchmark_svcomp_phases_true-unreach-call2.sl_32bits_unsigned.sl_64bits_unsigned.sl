(set-logic BV)

(synth-inv inv_fun ((x (_ BitVec 64))(y (_ BitVec 64))))

(define-fun pre_fun ((x (_ BitVec 64)) (y (_ BitVec 64))) Bool
       ( = x #x00000001 ))
(define-fun trans_fun ((x (_ BitVec 64)) (y (_ BitVec 64)) (x! (_ BitVec 64)) (y! (_ BitVec 64))) Bool
       ( let ( ( a!1 ( or ( and ( bvugt x #x00000000 ) ( bvult x ( bvudiv y x ) ) ( = x! ( bvmul x x ) ) ) ( = x! ( bvadd x #x00000001 ) ) ) ) ) ( and ( bvugt y #x00000000 ) ( bvult x y ) ( = y! y ) a!1 ) ))
(define-fun post_fun ((x (_ BitVec 64)) (y (_ BitVec 64))) Bool
       ( or ( = y #x00000000 ) ( = x y ) ( and ( bvult y #x00000000 ) ( bvult x y ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

