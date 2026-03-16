(set-logic BV)

(synth-inv inv_fun ((y (_ BitVec 64))(x (_ BitVec 64))(t (_ BitVec 64))(i (_ BitVec 64))))

(define-fun pre_fun ((y (_ BitVec 64)) (x (_ BitVec 64)) (t (_ BitVec 64)) (i (_ BitVec 64))) Bool
       ( and ( not ( = x y ) ) ( = i #x0000000000000000 ) ( = t y ) ))
(define-fun trans_fun ((y! (_ BitVec 64)) (x! (_ BitVec 64)) (t! (_ BitVec 64)) (i! (_ BitVec 64)) (y (_ BitVec 64)) (x (_ BitVec 64)) (t (_ BitVec 64)) (i (_ BitVec 64))) Bool
       ( let ( ( a!1 ( or ( and ( bvugt x #x0000000000000000 ) ( = i! i ) ( = t! t ) ( = x! x ) ( = y! ( bvadd x y ) ) ) ( and ( bvule x #x0000000000000000 ) ( = i! i ) ( = t! t ) ( = x! x ) ( = y! y ) ) ) ) ) ( and a!1 ) ))
(define-fun post_fun ((y (_ BitVec 64)) (x (_ BitVec 64)) (t (_ BitVec 64)) (i (_ BitVec 64))) Bool
       ( bvule t y ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

