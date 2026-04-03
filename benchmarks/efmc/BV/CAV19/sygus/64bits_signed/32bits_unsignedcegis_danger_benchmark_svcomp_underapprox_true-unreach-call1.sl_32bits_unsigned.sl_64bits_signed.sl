(set-logic BV)

(synth-inv inv_fun ((x (_ BitVec 64))(y (_ BitVec 64))))

(define-fun pre_fun ((x (_ BitVec 64)) (y (_ BitVec 64))) Bool
       ( and ( = x #x00000000 ) ( = y #x00000001 ) ))
(define-fun trans_fun ((x (_ BitVec 64)) (y (_ BitVec 64)) (x! (_ BitVec 64)) (y! (_ BitVec 64))) Bool
       ( and ( bvslt x #x00000006 ) ( = x! ( bvadd x #x00000001 ) ) ( = y! ( bvmul y #x00000002 ) ) ))
(define-fun post_fun ((x (_ BitVec 64)) (y (_ BitVec 64))) Bool
       ( or ( = #x00000001 ( bvurem y #x00000003 ) ) ( bvslt x #x00000006 ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

