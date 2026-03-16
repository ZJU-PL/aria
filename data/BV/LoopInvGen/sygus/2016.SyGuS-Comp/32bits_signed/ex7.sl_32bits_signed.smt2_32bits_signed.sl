(set-logic BV)

(synth-inv inv_fun ((i (_ BitVec 32))(y (_ BitVec 32))(x (_ BitVec 32))))

(define-fun pre_fun ((i (_ BitVec 32)) (y (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( and ( = i #x00000000 ) ( bvsge x #x00000000 ) ( bvsge y #x00000000 ) ( bvsge x y ) ))
(define-fun trans_fun ((i! (_ BitVec 32)) (y! (_ BitVec 32)) (x! (_ BitVec 32)) (i (_ BitVec 32)) (y (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( and ( bvslt i y ) ( = i! ( bvadd i #x00000001 ) ) ( = y! y ) ( = x! x ) ))
(define-fun post_fun ((i (_ BitVec 32)) (y (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( let ( ( a!1 ( and ( not ( bvsle y i ) ) ( or ( bvsle x i ) ( not ( bvsle #x00000000 i ) ) ) ) ) ) ( not a!1 ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

