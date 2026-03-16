(set-logic BV)

(synth-inv inv_fun ((n (_ BitVec 64))(m (_ BitVec 64))(x (_ BitVec 64))))

(define-fun pre_fun ((n (_ BitVec 64)) (m (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( and ( = x #x0000000000000000 ) ( = m #x0000000000000000 ) ))
(define-fun trans_fun ((n! (_ BitVec 64)) (m! (_ BitVec 64)) (x! (_ BitVec 64)) (n (_ BitVec 64)) (m (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( and ( bvslt x n ) ( or ( = m! x ) ( = m! m ) ) ( = n! n ) ( = x! ( bvadd x #x0000000000000001 ) ) ))
(define-fun post_fun ((n (_ BitVec 64)) (m (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( and ( or ( bvsle #x0000000000000000 m ) ( bvsle n #x0000000000000000 ) ) ( or ( not ( bvsle n m ) ) ( bvsle n #x0000000000000000 ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

