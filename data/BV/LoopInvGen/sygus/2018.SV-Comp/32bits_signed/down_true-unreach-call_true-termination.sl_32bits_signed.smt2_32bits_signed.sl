(set-logic BV)

(synth-inv inv_fun ((j (_ BitVec 32))(n (_ BitVec 32))(k (_ BitVec 32))(i (_ BitVec 32))))

(define-fun pre_fun ((j (_ BitVec 32)) (n (_ BitVec 32)) (k (_ BitVec 32)) (i (_ BitVec 32))) Bool
       ( and ( = k #x00000000 ) ( = i #x00000000 ) ( = j n ) ))
(define-fun trans_fun ((j! (_ BitVec 32)) (n! (_ BitVec 32)) (k! (_ BitVec 32)) (i! (_ BitVec 32)) (j (_ BitVec 32)) (n (_ BitVec 32)) (k (_ BitVec 32)) (i (_ BitVec 32))) Bool
       ( let ( ( a!1 ( or ( and ( bvslt i n ) ( = i! ( bvadd i #x00000001 ) ) ( = k! ( bvadd k #x00000001 ) ) ( = n! n ) ( = j! j ) ) ( and ( bvsge i n ) ( bvsgt j #x00000000 ) ( = j! ( bvsub j #x00000001 ) ) ( = k! ( bvsub k #x00000001 ) ) ) ( = n! n ) ( = i! i ) ) ) ) ( and a!1 ) ))
(define-fun post_fun ((j (_ BitVec 32)) (n (_ BitVec 32)) (k (_ BitVec 32)) (i (_ BitVec 32))) Bool
       ( let ( ( a!1 ( not ( and ( bvsle n i ) ( not ( bvsle j #x00000000 ) ) ) ) ) ) ( or a!1 ( not ( bvsle k #x00000000 ) ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

