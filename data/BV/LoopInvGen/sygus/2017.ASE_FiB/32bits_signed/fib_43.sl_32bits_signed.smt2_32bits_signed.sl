(set-logic BV)

(synth-inv inv_fun ((y (_ BitVec 32))(x (_ BitVec 32))(t (_ BitVec 32))(i (_ BitVec 32))))

(define-fun pre_fun ((y (_ BitVec 32)) (x (_ BitVec 32)) (t (_ BitVec 32)) (i (_ BitVec 32))) Bool
       ( and ( not ( = x y ) ) ( = i #x00000000 ) ( = t y ) ))
(define-fun trans_fun ((y! (_ BitVec 32)) (x! (_ BitVec 32)) (t! (_ BitVec 32)) (i! (_ BitVec 32)) (y (_ BitVec 32)) (x (_ BitVec 32)) (t (_ BitVec 32)) (i (_ BitVec 32))) Bool
       ( let ( ( a!1 ( or ( and ( bvsgt x #x00000000 ) ( = i! i ) ( = t! t ) ( = x! x ) ( = y! ( bvadd x y ) ) ) ( and ( bvsle x #x00000000 ) ( = i! i ) ( = t! t ) ( = x! x ) ( = y! y ) ) ) ) ) ( and a!1 ) ))
(define-fun post_fun ((y (_ BitVec 32)) (x (_ BitVec 32)) (t (_ BitVec 32)) (i (_ BitVec 32))) Bool
       ( bvsle t y ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

