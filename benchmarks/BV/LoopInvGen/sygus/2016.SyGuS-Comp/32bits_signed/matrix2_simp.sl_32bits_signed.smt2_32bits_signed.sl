(set-logic BV)

(synth-inv inv_fun ((c (_ BitVec 32))(r (_ BitVec 32))(k (_ BitVec 32))(j (_ BitVec 32))(a (_ BitVec 32))(m (_ BitVec 32))))

(define-fun pre_fun ((c (_ BitVec 32)) (r (_ BitVec 32)) (k (_ BitVec 32)) (j (_ BitVec 32)) (a (_ BitVec 32)) (m (_ BitVec 32))) Bool
       ( and ( or ( bvsle a m ) ( = j #x00000000 ) ) ( bvslt j r ) ( = k #x00000000 ) ))
(define-fun trans_fun ((c! (_ BitVec 32)) (r! (_ BitVec 32)) (k! (_ BitVec 32)) (j! (_ BitVec 32)) (a! (_ BitVec 32)) (m! (_ BitVec 32)) (c (_ BitVec 32)) (r (_ BitVec 32)) (k (_ BitVec 32)) (j (_ BitVec 32)) (a (_ BitVec 32)) (m (_ BitVec 32))) Bool
       ( let ( ( a!1 ( or ( and ( = j! j ) ( = r! r ) ( = c! c ) ( bvslt k c ) ( bvslt m a! ) ( = m! a! ) ( = k! ( bvadd k #x00000001 ) ) ) ( and ( = j! j ) ( = r! r ) ( = c! c ) ( bvslt k c ) ( bvsgt m a! ) ( = k! ( bvadd k #x00000001 ) ) ) ) ) ) ( and a!1 ) ))
(define-fun post_fun ((c (_ BitVec 32)) (r (_ BitVec 32)) (k (_ BitVec 32)) (j (_ BitVec 32)) (a (_ BitVec 32)) (m (_ BitVec 32))) Bool
       ( or ( bvsle a m ) ( not ( bvsle c k ) ) ( = j #xffffffff ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

