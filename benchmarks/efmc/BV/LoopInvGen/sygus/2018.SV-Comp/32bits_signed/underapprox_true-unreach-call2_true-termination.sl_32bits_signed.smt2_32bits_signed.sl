(set-logic BV)

(synth-inv inv_fun ((y (_ BitVec 32))(x (_ BitVec 32))))

(define-fun pre_fun ((y (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( and ( = x #x00000000 ) ( = y #x00000001 ) ))
(define-fun trans_fun ((y! (_ BitVec 32)) (x! (_ BitVec 32)) (y (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( and ( bvslt x #x00000006 ) ( = x! ( bvadd x #x00000001 ) ) ( = y! ( bvmul y #x00000002 ) ) ))
(define-fun post_fun ((y (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( or ( not ( bvsle #x00000006 x ) ) ( = x #x00000006 ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

