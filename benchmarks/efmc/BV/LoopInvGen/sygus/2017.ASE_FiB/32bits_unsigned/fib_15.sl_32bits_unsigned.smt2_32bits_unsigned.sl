(set-logic BV)

(synth-inv inv_fun ((k (_ BitVec 32))(j (_ BitVec 32))(n (_ BitVec 32))))

(define-fun pre_fun ((k (_ BitVec 32)) (j (_ BitVec 32)) (n (_ BitVec 32))) Bool
       ( and ( = j #x00000000 ) ( bvugt n #x00000000 ) ( bvugt k n ) ))
(define-fun trans_fun ((k! (_ BitVec 32)) (j! (_ BitVec 32)) (n! (_ BitVec 32)) (k (_ BitVec 32)) (j (_ BitVec 32)) (n (_ BitVec 32))) Bool
       ( let ( ( a!1 ( or ( and ( bvult j n ) ( = j! ( bvadd j #x00000001 ) ) ( = k! ( bvsub k #x00000001 ) ) ( = n! n ) ) ( and ( bvuge j n ) ( = j! j ) ( = k! k ) ( = n! n ) ) ) ) ) ( and a!1 ) ))
(define-fun post_fun ((k (_ BitVec 32)) (j (_ BitVec 32)) (n (_ BitVec 32))) Bool
       ( t r u e ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

