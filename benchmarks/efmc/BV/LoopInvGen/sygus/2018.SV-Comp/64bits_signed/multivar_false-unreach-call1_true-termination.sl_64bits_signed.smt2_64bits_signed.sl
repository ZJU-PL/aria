(set-logic BV)

(synth-inv inv_fun ((y (_ BitVec 64))(x (_ BitVec 64))))

(define-fun pre_fun ((y (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( = y ( bvadd x #x0000000000000001 ) ))
(define-fun trans_fun ((y! (_ BitVec 64)) (x! (_ BitVec 64)) (y (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( and ( bvslt x #x0000000000000400 ) ( = x! ( bvadd x #x0000000000000001 ) ) ( = y! ( bvadd y #x0000000000000001 ) ) ))
(define-fun post_fun ((y (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( or ( not ( bvsle #x0000000000000400 x ) ) ( not ( = x y ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

