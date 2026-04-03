(set-logic BV)

(synth-inv inv_fun ((j (_ BitVec 32))(i (_ BitVec 32))(k (_ BitVec 32))(n (_ BitVec 32))))

(define-fun pre_fun ((j (_ BitVec 32)) (i (_ BitVec 32)) (k (_ BitVec 32)) (n (_ BitVec 32))) Bool
       ( and ( = j #x00000000 ) ( bvsge n #x00000000 ) ( = i #x00000000 ) ( or ( = k #x00000001 ) ( bvsge k #x00000000 ) ) ))
(define-fun trans_fun ((j! (_ BitVec 32)) (i! (_ BitVec 32)) (k! (_ BitVec 32)) (n! (_ BitVec 32)) (j (_ BitVec 32)) (i (_ BitVec 32)) (k (_ BitVec 32)) (n (_ BitVec 32))) Bool
       ( let ( ( a!1 ( or ( and ( bvsgt i n ) ( = n! n ) ( = k! k ) ( = i! i ) ( = j! j ) ) ( and ( bvsle i n ) ( = n! n ) ( = k! k ) ( = i! ( bvadd i #x00000001 ) ) ( = j! ( bvadd j #x00000001 ) ) ) ) ) ) ( and a!1 ) ))
(define-fun post_fun ((j (_ BitVec 32)) (i (_ BitVec 32)) (k (_ BitVec 32)) (n (_ BitVec 32))) Bool
       ( or ( bvsle i n ) ( not ( bvsle ( bvadd k i j ) ( bvmul #x00000002 n ) ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

