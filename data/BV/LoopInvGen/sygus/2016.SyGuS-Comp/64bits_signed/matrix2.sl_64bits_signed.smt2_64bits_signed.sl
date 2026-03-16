(set-logic BV)

(synth-inv inv_fun ((k (_ BitVec 64))(j (_ BitVec 64))(a (_ BitVec 64))(m (_ BitVec 64))))

(define-fun pre_fun ((k (_ BitVec 64)) (j (_ BitVec 64)) (a (_ BitVec 64)) (m (_ BitVec 64))) Bool
       ( and ( or ( bvsle a m ) ( = j #x0000000000000000 ) ) ( bvslt j #x0000000000000001 ) ( = k #x0000000000000000 ) ))
(define-fun trans_fun ((k! (_ BitVec 64)) (j! (_ BitVec 64)) (a! (_ BitVec 64)) (m! (_ BitVec 64)) (k (_ BitVec 64)) (j (_ BitVec 64)) (a (_ BitVec 64)) (m (_ BitVec 64))) Bool
       ( let ( ( a!1 ( or ( and ( = j! j ) ( bvslt k #x0000000000000001 ) ( bvslt m a! ) ( = m! a! ) ( = k! ( bvadd k #x0000000000000001 ) ) ) ( and ( = j! j ) ( bvslt k #x0000000000000001 ) ( bvsgt m a! ) ( = k! ( bvadd k #x0000000000000001 ) ) ) ) ) ) ( and a!1 ) ))
(define-fun post_fun ((k (_ BitVec 64)) (j (_ BitVec 64)) (a (_ BitVec 64)) (m (_ BitVec 64))) Bool
       ( or ( not ( bvsle #x0000000000000001 k ) ) ( not ( bvsle m a ) ) ( = j #xffffffffffffffff ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

