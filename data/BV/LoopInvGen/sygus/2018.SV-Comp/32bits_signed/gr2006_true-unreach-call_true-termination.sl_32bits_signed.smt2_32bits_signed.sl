(set-logic BV)

(synth-inv inv_fun ((y (_ BitVec 32))(x (_ BitVec 32))))

(define-fun pre_fun ((y (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( and ( = x #x00000000 ) ( = y #x00000001 ) ))
(define-fun trans_fun ((y! (_ BitVec 32)) (x! (_ BitVec 32)) (y (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( and ( bvsge y #x00000000 ) ( = x! ( bvadd x #x00000001 ) ) ( ite ( bvslt x! #x00000032 ) ( = y! ( bvadd y #x00000001 ) ) ( = y! ( bvsub y #x00000001 ) ) ) ))
(define-fun post_fun ((y (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( or ( bvsle #x00000000 y ) ( = x #x00000064 ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

