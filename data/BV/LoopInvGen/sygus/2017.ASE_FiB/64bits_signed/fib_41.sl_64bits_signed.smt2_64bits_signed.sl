(set-logic BV)

(synth-inv inv_fun ((j (_ BitVec 64))(i (_ BitVec 64))(k (_ BitVec 64))(n (_ BitVec 64))))

(define-fun pre_fun ((j (_ BitVec 64)) (i (_ BitVec 64)) (k (_ BitVec 64)) (n (_ BitVec 64))) Bool
       ( and ( = j #x0000000000000000 ) ( bvsge n #x0000000000000000 ) ( = i #x0000000000000000 ) ( or ( = k #x0000000000000001 ) ( bvsge k #x0000000000000000 ) ) ))
(define-fun trans_fun ((j! (_ BitVec 64)) (i! (_ BitVec 64)) (k! (_ BitVec 64)) (n! (_ BitVec 64)) (j (_ BitVec 64)) (i (_ BitVec 64)) (k (_ BitVec 64)) (n (_ BitVec 64))) Bool
       ( let ( ( a!1 ( or ( and ( bvsgt i n ) ( = n! n ) ( = k! k ) ( = i! i ) ( = j! j ) ) ( and ( bvsle i n ) ( = n! n ) ( = k! k ) ( = i! ( bvadd i #x0000000000000001 ) ) ( = j! ( bvadd j #x0000000000000001 ) ) ) ) ) ) ( and a!1 ) ))
(define-fun post_fun ((j (_ BitVec 64)) (i (_ BitVec 64)) (k (_ BitVec 64)) (n (_ BitVec 64))) Bool
       ( or ( bvsle i n ) ( not ( bvsle ( bvadd k i j ) ( bvmul #x0000000000000002 n ) ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

