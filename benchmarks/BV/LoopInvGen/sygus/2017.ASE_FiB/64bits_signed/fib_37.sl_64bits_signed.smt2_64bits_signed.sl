(set-logic BV)

(synth-inv inv_fun ((n (_ BitVec 64))(m (_ BitVec 64))(x (_ BitVec 64))))

(define-fun pre_fun ((n (_ BitVec 64)) (m (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( and ( = x #x0000000000000000 ) ( = m #x0000000000000000 ) ( bvsgt n #x0000000000000000 ) ))
(define-fun trans_fun ((n! (_ BitVec 64)) (m! (_ BitVec 64)) (x! (_ BitVec 64)) (n (_ BitVec 64)) (m (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( let ( ( a!1 ( or ( and ( bvsge x n ) ( = x! x ) ( = m! m ) ( = n! n ) ) ( and ( bvslt x n ) ( = x! ( bvadd x #x0000000000000001 ) ) ( = m! x ) ( = n! n ) ) ( and ( bvslt x n ) ( = x! ( bvadd x #x0000000000000001 ) ) ( = m! m ) ( = n! n ) ) ) ) ) ( and a!1 ) ))
(define-fun post_fun ((n (_ BitVec 64)) (m (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( or ( bvsle n #x0000000000000000 ) ( and ( bvsle #x0000000000000000 m ) ( not ( bvsle n m ) ) ) ( not ( bvsle n x ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

