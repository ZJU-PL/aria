(set-logic BV)

(synth-inv inv_fun ((n (_ BitVec 32))(m (_ BitVec 32))(x (_ BitVec 32))))

(define-fun pre_fun ((n (_ BitVec 32)) (m (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( and ( = x #x00000000 ) ( = m #x00000000 ) ( bvugt n #x00000000 ) ))
(define-fun trans_fun ((n! (_ BitVec 32)) (m! (_ BitVec 32)) (x! (_ BitVec 32)) (n (_ BitVec 32)) (m (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( let ( ( a!1 ( or ( and ( bvuge x n ) ( = x! x ) ( = m! m ) ( = n! n ) ) ( and ( bvult x n ) ( = x! ( bvadd x #x00000001 ) ) ( = m! x ) ( = n! n ) ) ( and ( bvult x n ) ( = x! ( bvadd x #x00000001 ) ) ( = m! m ) ( = n! n ) ) ) ) ) ( and a!1 ) ))
(define-fun post_fun ((n (_ BitVec 32)) (m (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( or ( = n #x00000000 ) ( not ( bvule n m ) ) ( not ( bvule n x ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

