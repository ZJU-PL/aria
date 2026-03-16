(set-logic BV)

(synth-inv inv_fun ((k (_ BitVec 64))(j (_ BitVec 64))(n (_ BitVec 64))))

(define-fun pre_fun ((k (_ BitVec 64)) (j (_ BitVec 64)) (n (_ BitVec 64))) Bool
       ( and ( = j #x0000000000000000 ) ( bvsgt n #x0000000000000000 ) ( bvsgt k n ) ))
(define-fun trans_fun ((k! (_ BitVec 64)) (j! (_ BitVec 64)) (n! (_ BitVec 64)) (k (_ BitVec 64)) (j (_ BitVec 64)) (n (_ BitVec 64))) Bool
       ( let ( ( a!1 ( or ( and ( bvslt j n ) ( = j! ( bvadd j #x0000000000000001 ) ) ( = k! ( bvsub k #x0000000000000001 ) ) ( = n! n ) ) ( and ( bvsge j n ) ( = j! j ) ( = k! k ) ( = n! n ) ) ) ) ) ( and a!1 ) ))
(define-fun post_fun ((k (_ BitVec 64)) (j (_ BitVec 64)) (n (_ BitVec 64))) Bool
       ( or ( not ( bvsle n j ) ) ( bvsle #x0000000000000000 k ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

