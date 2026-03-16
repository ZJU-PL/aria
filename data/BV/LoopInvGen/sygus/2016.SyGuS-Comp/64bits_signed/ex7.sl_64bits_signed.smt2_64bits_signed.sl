(set-logic BV)

(synth-inv inv_fun ((i (_ BitVec 64))(y (_ BitVec 64))(x (_ BitVec 64))))

(define-fun pre_fun ((i (_ BitVec 64)) (y (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( and ( = i #x0000000000000000 ) ( bvsge x #x0000000000000000 ) ( bvsge y #x0000000000000000 ) ( bvsge x y ) ))
(define-fun trans_fun ((i! (_ BitVec 64)) (y! (_ BitVec 64)) (x! (_ BitVec 64)) (i (_ BitVec 64)) (y (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( and ( bvslt i y ) ( = i! ( bvadd i #x0000000000000001 ) ) ( = y! y ) ( = x! x ) ))
(define-fun post_fun ((i (_ BitVec 64)) (y (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( let ( ( a!1 ( and ( not ( bvsle y i ) ) ( or ( bvsle x i ) ( not ( bvsle #x0000000000000000 i ) ) ) ) ) ) ( not a!1 ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

