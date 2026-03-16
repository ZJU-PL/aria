(set-logic BV)

(synth-inv inv_fun ((y (_ BitVec 32))(x (_ BitVec 32))))

(define-fun pre_fun ((y (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( and ( = x #x00000000 ) ( = y #x00000001 ) ))
(define-fun trans_fun ((y! (_ BitVec 32)) (x! (_ BitVec 32)) (y (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( let ( ( a!1 ( and ( bvsgt x! ( bvadd #x00000001 ( bvnot #x00000064 ) ) ) ( bvslt x! #x00000064 ) ) ) ) ( let ( ( a!2 ( = y! ( ite a!1 y ( bvadd #x00000001 ( bvnot y ) ) ) ) ) ) ( and ( = x! ( bvadd x y ) ) a!2 ) ) ))
(define-fun post_fun ((y (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( not ( and ( bvsle #xffffff9c x ) ( bvsle x #x00000064 ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

