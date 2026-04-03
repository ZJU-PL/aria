(set-logic BV)

(synth-inv inv_fun ((k (_ BitVec 64))(j (_ BitVec 64))(a (_ BitVec 64))(m (_ BitVec 64))))

(define-fun pre_fun ((k (_ BitVec 64)) (j (_ BitVec 64)) (a (_ BitVec 64)) (m (_ BitVec 64))) Bool
       ( and ( or ( bvule a m ) ( = j #x0000000000000000 ) ) ( bvult j #x0000000000000001 ) ( = k #x0000000000000000 ) ))
(define-fun trans_fun ((k! (_ BitVec 64)) (j! (_ BitVec 64)) (a! (_ BitVec 64)) (m! (_ BitVec 64)) (k (_ BitVec 64)) (j (_ BitVec 64)) (a (_ BitVec 64)) (m (_ BitVec 64))) Bool
       ( let ( ( a!1 ( or ( and ( = j! j ) ( bvult k #x0000000000000001 ) ( bvult m a! ) ( = m! a! ) ( = k! ( bvadd k #x0000000000000001 ) ) ) ( and ( = j! j ) ( bvult k #x0000000000000001 ) ( bvugt m a! ) ( = k! ( bvadd k #x0000000000000001 ) ) ) ) ) ) ( and a!1 ) ))
(define-fun post_fun ((k (_ BitVec 64)) (j (_ BitVec 64)) (a (_ BitVec 64)) (m (_ BitVec 64))) Bool
       ( or ( not ( bvule #x0000000000000001 k ) ) ( not ( bvule m a ) ) ( = j #xffffffffffffffff ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

