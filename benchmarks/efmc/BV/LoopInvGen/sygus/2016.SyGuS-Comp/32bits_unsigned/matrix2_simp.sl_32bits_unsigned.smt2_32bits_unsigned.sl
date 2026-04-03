(set-logic BV)

(synth-inv inv_fun ((c (_ BitVec 32))(r (_ BitVec 32))(k (_ BitVec 32))(j (_ BitVec 32))(a (_ BitVec 32))(m (_ BitVec 32))))

(define-fun pre_fun ((c (_ BitVec 32)) (r (_ BitVec 32)) (k (_ BitVec 32)) (j (_ BitVec 32)) (a (_ BitVec 32)) (m (_ BitVec 32))) Bool
       ( and ( or ( bvule a m ) ( = j #x00000000 ) ) ( bvult j r ) ( = k #x00000000 ) ))
(define-fun trans_fun ((c! (_ BitVec 32)) (r! (_ BitVec 32)) (k! (_ BitVec 32)) (j! (_ BitVec 32)) (a! (_ BitVec 32)) (m! (_ BitVec 32)) (c (_ BitVec 32)) (r (_ BitVec 32)) (k (_ BitVec 32)) (j (_ BitVec 32)) (a (_ BitVec 32)) (m (_ BitVec 32))) Bool
       ( let ( ( a!1 ( or ( and ( = j! j ) ( = r! r ) ( = c! c ) ( bvult k c ) ( bvult m a! ) ( = m! a! ) ( = k! ( bvadd k #x00000001 ) ) ) ( and ( = j! j ) ( = r! r ) ( = c! c ) ( bvult k c ) ( bvugt m a! ) ( = k! ( bvadd k #x00000001 ) ) ) ) ) ) ( and a!1 ) ))
(define-fun post_fun ((c (_ BitVec 32)) (r (_ BitVec 32)) (k (_ BitVec 32)) (j (_ BitVec 32)) (a (_ BitVec 32)) (m (_ BitVec 32))) Bool
       ( or ( = j #xffffffff ) ( bvule a m ) ( not ( bvule c k ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

