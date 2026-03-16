(set-logic BV)

(synth-inv inv_fun ((y (_ BitVec 32))(x (_ BitVec 32))))

(define-fun pre_fun ((y (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( = x ( bvadd #x00000001 ( bvnot #x00000032 ) ) ))
(define-fun trans_fun ((y! (_ BitVec 32)) (x! (_ BitVec 32)) (y (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( let ( ( a!1 ( or ( and ( bvslt x #x00000000 ) ( = x! ( bvadd x y ) ) ( = y! ( bvadd y #x00000001 ) ) ) ( and ( bvsge x #x00000000 ) ( = x! x ) ( = y! y ) ) ) ) ) ( and a!1 ) ))
(define-fun post_fun ((y (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( or ( not ( bvsle #x00000000 x ) ) ( bvsle #x00000000 y ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

