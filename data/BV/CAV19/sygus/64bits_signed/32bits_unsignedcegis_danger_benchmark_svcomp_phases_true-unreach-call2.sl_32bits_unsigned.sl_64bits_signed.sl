(set-logic BV)

(synth-inv inv_fun ((x (_ BitVec 64))(y (_ BitVec 64))))

(define-fun pre_fun ((x (_ BitVec 64)) (y (_ BitVec 64))) Bool
       ( = x #x00000001 ))
(define-fun trans_fun ((x (_ BitVec 64)) (y (_ BitVec 64)) (x! (_ BitVec 64)) (y! (_ BitVec 64))) Bool
       ( let ( ( a!1 ( or ( and ( bvsgt x #x00000000 ) ( bvslt x ( bvudiv y x ) ) ( = x! ( bvmul x x ) ) ) ( = x! ( bvadd x #x00000001 ) ) ) ) ) ( and ( bvsgt y #x00000000 ) ( bvslt x y ) ( = y! y ) a!1 ) ))
(define-fun post_fun ((x (_ BitVec 64)) (y (_ BitVec 64))) Bool
       ( or ( = y #x00000000 ) ( = x y ) ( and ( bvslt y #x00000000 ) ( bvslt x y ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

