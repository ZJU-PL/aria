(set-logic BV)

(synth-inv inv_fun ((c (_ BitVec 64))(r (_ BitVec 64))(k (_ BitVec 64))(j (_ BitVec 64))(a (_ BitVec 64))(m (_ BitVec 64))))

(define-fun pre_fun ((c (_ BitVec 64)) (r (_ BitVec 64)) (k (_ BitVec 64)) (j (_ BitVec 64)) (a (_ BitVec 64)) (m (_ BitVec 64))) Bool
       ( and ( or ( bvsle a m ) ( = j #x0000000000000000 ) ) ( bvslt j r ) ( = k #x0000000000000000 ) ))
(define-fun trans_fun ((c! (_ BitVec 64)) (r! (_ BitVec 64)) (k! (_ BitVec 64)) (j! (_ BitVec 64)) (a! (_ BitVec 64)) (m! (_ BitVec 64)) (c (_ BitVec 64)) (r (_ BitVec 64)) (k (_ BitVec 64)) (j (_ BitVec 64)) (a (_ BitVec 64)) (m (_ BitVec 64))) Bool
       ( let ( ( a!1 ( or ( and ( = j! j ) ( = r! r ) ( = c! c ) ( bvslt k c ) ( bvslt m a! ) ( = m! a! ) ( = k! ( bvadd k #x0000000000000001 ) ) ) ( and ( = j! j ) ( = r! r ) ( = c! c ) ( bvslt k c ) ( bvsgt m a! ) ( = k! ( bvadd k #x0000000000000001 ) ) ) ) ) ) ( and a!1 ) ))
(define-fun post_fun ((c (_ BitVec 64)) (r (_ BitVec 64)) (k (_ BitVec 64)) (j (_ BitVec 64)) (a (_ BitVec 64)) (m (_ BitVec 64))) Bool
       ( or ( not ( bvsle c k ) ) ( = j #xffffffffffffffff ) ( bvsle a m ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

