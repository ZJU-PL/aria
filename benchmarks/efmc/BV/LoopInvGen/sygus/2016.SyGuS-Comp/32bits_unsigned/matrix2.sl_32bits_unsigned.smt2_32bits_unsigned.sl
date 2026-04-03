(set-logic BV)

(synth-inv inv_fun ((k (_ BitVec 32))(j (_ BitVec 32))(a (_ BitVec 32))(m (_ BitVec 32))))

(define-fun pre_fun ((k (_ BitVec 32)) (j (_ BitVec 32)) (a (_ BitVec 32)) (m (_ BitVec 32))) Bool
       ( and ( or ( bvule a m ) ( = j #x00000000 ) ) ( bvult j #x00000001 ) ( = k #x00000000 ) ))
(define-fun trans_fun ((k! (_ BitVec 32)) (j! (_ BitVec 32)) (a! (_ BitVec 32)) (m! (_ BitVec 32)) (k (_ BitVec 32)) (j (_ BitVec 32)) (a (_ BitVec 32)) (m (_ BitVec 32))) Bool
       ( let ( ( a!1 ( or ( and ( = j! j ) ( bvult k #x00000001 ) ( bvult m a! ) ( = m! a! ) ( = k! ( bvadd k #x00000001 ) ) ) ( and ( = j! j ) ( bvult k #x00000001 ) ( bvugt m a! ) ( = k! ( bvadd k #x00000001 ) ) ) ) ) ) ( and a!1 ) ))
(define-fun post_fun ((k (_ BitVec 32)) (j (_ BitVec 32)) (a (_ BitVec 32)) (m (_ BitVec 32))) Bool
       ( or ( not ( bvule #x00000001 k ) ) ( not ( bvule m a ) ) ( = j #xffffffff ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

