(set-logic BV)

(synth-inv inv_fun ((y (_ BitVec 32))(x (_ BitVec 32))))

(define-fun pre_fun ((y (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( and ( = x ( bvsub #x00000000 #x00000032 ) ) ( bvslt ( bvsub #x00000000 #x000003e8 ) y ) ))
(define-fun trans_fun ((y! (_ BitVec 32)) (x! (_ BitVec 32)) (y (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( and ( bvslt x #x00000000 ) ( = x! ( bvadd x y ) ) ( = y! ( bvadd y #x00000001 ) ) ))
(define-fun post_fun ((y (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( or ( not ( bvsle #x00000000 x ) ) ( not ( bvsle y #x00000000 ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

