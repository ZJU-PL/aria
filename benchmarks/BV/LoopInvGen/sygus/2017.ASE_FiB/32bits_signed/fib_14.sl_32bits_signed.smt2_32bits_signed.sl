(set-logic BV)

(synth-inv inv_fun ((m (_ BitVec 32))(j (_ BitVec 32))(a (_ BitVec 32))))

(define-fun pre_fun ((m (_ BitVec 32)) (j (_ BitVec 32)) (a (_ BitVec 32))) Bool
       ( and ( = a #x00000000 ) ( bvsgt m #x00000000 ) ( = j #x00000001 ) ))
(define-fun trans_fun ((m! (_ BitVec 32)) (j! (_ BitVec 32)) (a! (_ BitVec 32)) (m (_ BitVec 32)) (j (_ BitVec 32)) (a (_ BitVec 32))) Bool
       ( let ( ( a!1 ( or ( and ( bvsgt j m ) ( = a! a ) ( = j! j ) ( = m! m ) ) ( and ( bvsle j m ) ( = j! ( bvadd j #x00000001 ) ) ( = a! ( bvadd a #x00000001 ) ) ( = m! m ) ) ( and ( bvsle j m ) ( = j! ( bvadd j #x00000001 ) ) ( = a! ( bvsub a #x00000001 ) ) ( = m! m ) ) ) ) ) ( and a!1 ) ))
(define-fun post_fun ((m (_ BitVec 32)) (j (_ BitVec 32)) (a (_ BitVec 32))) Bool
       ( or ( bvsle j m ) ( and ( bvsle ( bvmul #xffffffff m ) a ) ( bvsle a m ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

