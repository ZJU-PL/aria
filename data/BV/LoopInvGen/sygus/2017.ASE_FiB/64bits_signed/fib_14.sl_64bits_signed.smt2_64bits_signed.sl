(set-logic BV)

(synth-inv inv_fun ((m (_ BitVec 64))(j (_ BitVec 64))(a (_ BitVec 64))))

(define-fun pre_fun ((m (_ BitVec 64)) (j (_ BitVec 64)) (a (_ BitVec 64))) Bool
       ( and ( = a #x0000000000000000 ) ( bvsgt m #x0000000000000000 ) ( = j #x0000000000000001 ) ))
(define-fun trans_fun ((m! (_ BitVec 64)) (j! (_ BitVec 64)) (a! (_ BitVec 64)) (m (_ BitVec 64)) (j (_ BitVec 64)) (a (_ BitVec 64))) Bool
       ( let ( ( a!1 ( or ( and ( bvsgt j m ) ( = a! a ) ( = j! j ) ( = m! m ) ) ( and ( bvsle j m ) ( = j! ( bvadd j #x0000000000000001 ) ) ( = a! ( bvadd a #x0000000000000001 ) ) ( = m! m ) ) ( and ( bvsle j m ) ( = j! ( bvadd j #x0000000000000001 ) ) ( = a! ( bvsub a #x0000000000000001 ) ) ( = m! m ) ) ) ) ) ( and a!1 ) ))
(define-fun post_fun ((m (_ BitVec 64)) (j (_ BitVec 64)) (a (_ BitVec 64))) Bool
       ( or ( bvsle j m ) ( and ( bvsle ( bvmul #xffffffffffffffff m ) a ) ( bvsle a m ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

