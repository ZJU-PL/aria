(set-logic BV)

(synth-inv inv_fun ((y (_ BitVec 64))(x (_ BitVec 64))))

(define-fun pre_fun ((y (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( and ( = x #x0000000000000000 ) ( = y #x0000000000000032 ) ))
(define-fun trans_fun ((y! (_ BitVec 64)) (x! (_ BitVec 64)) (y (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( and ( bvslt x #x0000000000000064 ) ( ite ( bvslt x #x0000000000000032 ) ( = x! ( bvadd x #x0000000000000001 ) ) ( = y! y ) ) ( = x! ( bvadd x #x0000000000000001 ) ) ( = y! ( bvadd y #x0000000000000001 ) ) ))
(define-fun post_fun ((y (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( or ( not ( bvsle #x0000000000000064 x ) ) ( = y #x0000000000000064 ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

