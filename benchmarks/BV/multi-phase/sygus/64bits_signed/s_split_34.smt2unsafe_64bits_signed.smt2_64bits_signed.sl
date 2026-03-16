(set-logic BV)

(synth-inv inv_fun ((y (_ BitVec 64))(x (_ BitVec 64))))

(define-fun pre_fun ((y (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( and ( = x #x0000000000000000 ) ( = y #x0000000000000001 ) ))
(define-fun trans_fun ((y! (_ BitVec 64)) (x! (_ BitVec 64)) (y (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( let ( ( a!1 ( and ( bvsgt x! ( bvadd #x0000000000000001 ( bvnot #x0000000000000064 ) ) ) ( bvslt x! #x0000000000000064 ) ) ) ) ( let ( ( a!2 ( = y! ( ite a!1 y ( bvadd #x0000000000000001 ( bvnot y ) ) ) ) ) ) ( and ( = x! ( bvadd x y ) ) a!2 ) ) ))
(define-fun post_fun ((y (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( not ( and ( bvsle #xffffffffffffff9c x ) ( bvsle x #x0000000000000064 ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

