(set-logic BV)

(synth-inv inv_fun ((y (_ BitVec 64))(x (_ BitVec 64))))

(define-fun pre_fun ((y (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( and ( = x #x0000000000000000 ) ( = y #x0000000000000001 ) ))
(define-fun trans_fun ((y! (_ BitVec 64)) (x! (_ BitVec 64)) (y (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( and ( bvsge y #x0000000000000000 ) ( = x! ( bvadd x #x0000000000000001 ) ) ( ite ( bvslt x! #x0000000000000032 ) ( = y! ( bvadd y #x0000000000000001 ) ) ( = y! ( bvsub y #x0000000000000001 ) ) ) ))
(define-fun post_fun ((y (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( or ( = x #x0000000000000064 ) ( bvsle #x0000000000000000 y ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

