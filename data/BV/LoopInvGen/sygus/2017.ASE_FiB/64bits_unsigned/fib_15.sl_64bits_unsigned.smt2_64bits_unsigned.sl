(set-logic BV)

(synth-inv inv_fun ((k (_ BitVec 64))(j (_ BitVec 64))(n (_ BitVec 64))))

(define-fun pre_fun ((k (_ BitVec 64)) (j (_ BitVec 64)) (n (_ BitVec 64))) Bool
       ( and ( = j #x0000000000000000 ) ( bvugt n #x0000000000000000 ) ( bvugt k n ) ))
(define-fun trans_fun ((k! (_ BitVec 64)) (j! (_ BitVec 64)) (n! (_ BitVec 64)) (k (_ BitVec 64)) (j (_ BitVec 64)) (n (_ BitVec 64))) Bool
       ( let ( ( a!1 ( or ( and ( bvult j n ) ( = j! ( bvadd j #x0000000000000001 ) ) ( = k! ( bvsub k #x0000000000000001 ) ) ( = n! n ) ) ( and ( bvuge j n ) ( = j! j ) ( = k! k ) ( = n! n ) ) ) ) ) ( and a!1 ) ))
(define-fun post_fun ((k (_ BitVec 64)) (j (_ BitVec 64)) (n (_ BitVec 64))) Bool
       ( t r u e ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

