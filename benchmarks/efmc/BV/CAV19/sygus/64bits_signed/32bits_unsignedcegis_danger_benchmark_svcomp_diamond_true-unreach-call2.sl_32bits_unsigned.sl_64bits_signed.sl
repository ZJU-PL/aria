(set-logic BV)

(synth-inv inv_fun ((x (_ BitVec 64))(y (_ BitVec 64))))

(define-fun pre_fun ((x (_ BitVec 64)) (y (_ BitVec 64))) Bool
       ( = x #x00000000 ))
(define-fun trans_fun ((x (_ BitVec 64)) (y (_ BitVec 64)) (x! (_ BitVec 64)) (y! (_ BitVec 64))) Bool
       ( or ( and ( = ( bvurem y #x00000002 ) #x00000000 ) ( = x! ( bvadd x #x00000014 ) ) ( and ( = y! y ) ( bvslt x #x00000063 ) ) ) ( and ( = x! ( bvsub #x00000005 x ) ) ( and ( = y! y ) ( bvslt x #x00000063 ) ) ) ))
(define-fun post_fun ((x (_ BitVec 64)) (y (_ BitVec 64))) Bool
       ( or ( = ( bvurem x #x00000002 ) ( bvurem y #x00000002 ) ) ( bvslt x #x00000063 ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

