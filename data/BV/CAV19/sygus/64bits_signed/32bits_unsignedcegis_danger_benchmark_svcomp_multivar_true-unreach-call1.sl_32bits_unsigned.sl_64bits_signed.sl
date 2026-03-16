(set-logic BV)

(synth-inv inv_fun ((x (_ BitVec 64))(y (_ BitVec 64))))

(define-fun pre_fun ((x (_ BitVec 64)) (y (_ BitVec 64))) Bool
       ( = x y ))
(define-fun trans_fun ((x (_ BitVec 64)) (x! (_ BitVec 64)) (y (_ BitVec 64)) (y! (_ BitVec 64))) Bool
       ( and ( bvslt x #x00000400 ) ( = x! ( bvadd x #x00000001 ) ) ( = y! ( bvadd y #x00000001 ) ) ))
(define-fun post_fun ((x (_ BitVec 64)) (y (_ BitVec 64))) Bool
       ( or ( = x y ) ( bvslt x #x00000400 ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

