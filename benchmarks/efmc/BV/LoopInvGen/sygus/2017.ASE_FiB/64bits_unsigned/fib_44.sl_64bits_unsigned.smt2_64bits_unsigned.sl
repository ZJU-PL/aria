(set-logic BV)

(synth-inv inv_fun ((k (_ BitVec 64))(n (_ BitVec 64))(j (_ BitVec 64))(i (_ BitVec 64))))

(define-fun pre_fun ((k (_ BitVec 64)) (n (_ BitVec 64)) (j (_ BitVec 64)) (i (_ BitVec 64))) Bool
       ( and ( or ( = n #x0000000000000001 ) ( = n #x0000000000000002 ) ) ( = i #x0000000000000000 ) ( = j #x0000000000000000 ) ))
(define-fun trans_fun ((k! (_ BitVec 64)) (n! (_ BitVec 64)) (j! (_ BitVec 64)) (i! (_ BitVec 64)) (k (_ BitVec 64)) (n (_ BitVec 64)) (j (_ BitVec 64)) (i (_ BitVec 64))) Bool
       ( let ( ( a!1 ( or ( and ( bvugt i k ) ( = i! i ) ( = j! j ) ( = n! n ) ( = k! k ) ) ( and ( bvule i k ) ( = i! ( bvadd i #x0000000000000001 ) ) ( = j! ( bvadd j n ) ) ( = n! n ) ( = k! k ) ) ) ) ) ( and a!1 ) ))
(define-fun post_fun ((k (_ BitVec 64)) (n (_ BitVec 64)) (j (_ BitVec 64)) (i (_ BitVec 64))) Bool
       ( or ( not ( = n #x0000000000000001 ) ) ( bvule i k ) ( = i j ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

