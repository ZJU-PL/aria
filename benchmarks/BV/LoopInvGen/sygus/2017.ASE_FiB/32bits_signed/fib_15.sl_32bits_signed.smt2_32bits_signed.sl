(set-logic BV)

(synth-inv inv_fun ((k (_ BitVec 32))(j (_ BitVec 32))(n (_ BitVec 32))))

(define-fun pre_fun ((k (_ BitVec 32)) (j (_ BitVec 32)) (n (_ BitVec 32))) Bool
       ( and ( = j #x00000000 ) ( bvsgt n #x00000000 ) ( bvsgt k n ) ))
(define-fun trans_fun ((k! (_ BitVec 32)) (j! (_ BitVec 32)) (n! (_ BitVec 32)) (k (_ BitVec 32)) (j (_ BitVec 32)) (n (_ BitVec 32))) Bool
       ( let ( ( a!1 ( or ( and ( bvslt j n ) ( = j! ( bvadd j #x00000001 ) ) ( = k! ( bvsub k #x00000001 ) ) ( = n! n ) ) ( and ( bvsge j n ) ( = j! j ) ( = k! k ) ( = n! n ) ) ) ) ) ( and a!1 ) ))
(define-fun post_fun ((k (_ BitVec 32)) (j (_ BitVec 32)) (n (_ BitVec 32))) Bool
       ( or ( not ( bvsle n j ) ) ( bvsle #x00000000 k ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

