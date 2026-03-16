(set-logic BV)

(synth-inv inv_fun ((n (_ BitVec 32))(sum (_ BitVec 32))(i (_ BitVec 32))))

(define-fun pre_fun ((n (_ BitVec 32)) (sum (_ BitVec 32)) (i (_ BitVec 32))) Bool
       ( and ( = sum #x00000000 ) ( bvsge n #x00000000 ) ( = i #x00000000 ) ))
(define-fun trans_fun ((n! (_ BitVec 32)) (sum! (_ BitVec 32)) (i! (_ BitVec 32)) (n (_ BitVec 32)) (sum (_ BitVec 32)) (i (_ BitVec 32))) Bool
       ( let ( ( a!1 ( or ( and ( bvslt i n ) ( = i! ( bvadd i #x00000001 ) ) ( = sum! ( bvadd sum i ) ) ( = n! n ) ) ( and ( bvsge i n ) ( = i! i ) ( = sum! sum ) ( = n! n ) ) ) ) ) ( and a!1 ) ))
(define-fun post_fun ((n (_ BitVec 32)) (sum (_ BitVec 32)) (i (_ BitVec 32))) Bool
       ( or ( not ( bvsle n i ) ) ( bvsle #x00000000 sum ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

