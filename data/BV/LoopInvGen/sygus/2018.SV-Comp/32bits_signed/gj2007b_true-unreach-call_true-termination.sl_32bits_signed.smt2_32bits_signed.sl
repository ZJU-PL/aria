(set-logic BV)

(synth-inv inv_fun ((n (_ BitVec 32))(m (_ BitVec 32))(x (_ BitVec 32))))

(define-fun pre_fun ((n (_ BitVec 32)) (m (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( and ( = x #x00000000 ) ( = m #x00000000 ) ))
(define-fun trans_fun ((n! (_ BitVec 32)) (m! (_ BitVec 32)) (x! (_ BitVec 32)) (n (_ BitVec 32)) (m (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( and ( bvslt x n ) ( or ( = m! x ) ( = m! m ) ) ( = n! n ) ( = x! ( bvadd x #x00000001 ) ) ))
(define-fun post_fun ((n (_ BitVec 32)) (m (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( and ( or ( bvsle #x00000000 m ) ( bvsle n #x00000000 ) ) ( or ( not ( bvsle n m ) ) ( bvsle n #x00000000 ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

