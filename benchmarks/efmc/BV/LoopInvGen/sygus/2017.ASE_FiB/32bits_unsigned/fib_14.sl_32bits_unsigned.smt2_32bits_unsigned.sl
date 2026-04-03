(set-logic BV)

(synth-inv inv_fun ((m (_ BitVec 32))(j (_ BitVec 32))(a (_ BitVec 32))))

(define-fun pre_fun ((m (_ BitVec 32)) (j (_ BitVec 32)) (a (_ BitVec 32))) Bool
       ( and ( = a #x00000000 ) ( bvugt m #x00000000 ) ( = j #x00000001 ) ))
(define-fun trans_fun ((m! (_ BitVec 32)) (j! (_ BitVec 32)) (a! (_ BitVec 32)) (m (_ BitVec 32)) (j (_ BitVec 32)) (a (_ BitVec 32))) Bool
       ( let ( ( a!1 ( or ( and ( bvugt j m ) ( = a! a ) ( = j! j ) ( = m! m ) ) ( and ( bvule j m ) ( = j! ( bvadd j #x00000001 ) ) ( = a! ( bvadd a #x00000001 ) ) ( = m! m ) ) ( and ( bvule j m ) ( = j! ( bvadd j #x00000001 ) ) ( = a! ( bvsub a #x00000001 ) ) ( = m! m ) ) ) ) ) ( and a!1 ) ))
(define-fun post_fun ((m (_ BitVec 32)) (j (_ BitVec 32)) (a (_ BitVec 32))) Bool
       ( or ( bvule j m ) ( and ( bvule ( bvmul #xffffffff m ) a ) ( bvule a m ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

