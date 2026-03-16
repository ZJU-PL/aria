(set-logic BV)

(synth-inv inv_fun ((k (_ BitVec 32))(n (_ BitVec 32))(j (_ BitVec 32))(i (_ BitVec 32))))

(define-fun pre_fun ((k (_ BitVec 32)) (n (_ BitVec 32)) (j (_ BitVec 32)) (i (_ BitVec 32))) Bool
       ( and ( or ( = n #x00000001 ) ( = n #x00000002 ) ) ( = i #x00000000 ) ( = j #x00000000 ) ))
(define-fun trans_fun ((k! (_ BitVec 32)) (n! (_ BitVec 32)) (j! (_ BitVec 32)) (i! (_ BitVec 32)) (k (_ BitVec 32)) (n (_ BitVec 32)) (j (_ BitVec 32)) (i (_ BitVec 32))) Bool
       ( let ( ( a!1 ( or ( and ( bvsgt i k ) ( = i! i ) ( = j! j ) ( = n! n ) ( = k! k ) ) ( and ( bvsle i k ) ( = i! ( bvadd i #x00000001 ) ) ( = j! ( bvadd j n ) ) ( = n! n ) ( = k! k ) ) ) ) ) ( and a!1 ) ))
(define-fun post_fun ((k (_ BitVec 32)) (n (_ BitVec 32)) (j (_ BitVec 32)) (i (_ BitVec 32))) Bool
       ( or ( not ( = n #x00000001 ) ) ( bvsle i k ) ( = i j ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

