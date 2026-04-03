(set-logic BV)

(synth-inv inv_fun ((y (_ BitVec 64))(x (_ BitVec 64))))

(define-fun pre_fun ((y (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( and ( = x #x0000000000000000 ) ( = y #x0000000000000001 ) ))
(define-fun trans_fun ((y! (_ BitVec 64)) (x! (_ BitVec 64)) (y (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( and ( bvslt x #x0000000000000006 ) ( = x! ( bvadd x #x0000000000000001 ) ) ( = y! ( bvmul y #x0000000000000002 ) ) ))
(define-fun post_fun ((y (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( or ( not ( = y #x0000000000000040 ) ) ( not ( bvsle #x0000000000000006 x ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

