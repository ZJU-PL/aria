(set-logic BV)

(synth-inv inv_fun ((n (_ BitVec 64))(m (_ BitVec 64))(x (_ BitVec 64))))

(define-fun pre_fun ((n (_ BitVec 64)) (m (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( and ( = x #x0000000000000000 ) ( = m #x0000000000000000 ) ( bvugt n #x0000000000000000 ) ))
(define-fun trans_fun ((n! (_ BitVec 64)) (m! (_ BitVec 64)) (x! (_ BitVec 64)) (n (_ BitVec 64)) (m (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( let ( ( a!1 ( or ( and ( bvuge x n ) ( = x! x ) ( = m! m ) ( = n! n ) ) ( and ( bvult x n ) ( = x! ( bvadd x #x0000000000000001 ) ) ( = m! x ) ( = n! n ) ) ( and ( bvult x n ) ( = x! ( bvadd x #x0000000000000001 ) ) ( = m! m ) ( = n! n ) ) ) ) ) ( and a!1 ) ))
(define-fun post_fun ((n (_ BitVec 64)) (m (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( or ( not ( bvule n m ) ) ( not ( bvule n x ) ) ( = n #x0000000000000000 ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

