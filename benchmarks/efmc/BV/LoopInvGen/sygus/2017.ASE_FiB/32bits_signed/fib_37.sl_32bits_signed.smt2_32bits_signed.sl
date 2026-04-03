(set-logic BV)

(synth-inv inv_fun ((n (_ BitVec 32))(m (_ BitVec 32))(x (_ BitVec 32))))

(define-fun pre_fun ((n (_ BitVec 32)) (m (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( and ( = x #x00000000 ) ( = m #x00000000 ) ( bvsgt n #x00000000 ) ))
(define-fun trans_fun ((n! (_ BitVec 32)) (m! (_ BitVec 32)) (x! (_ BitVec 32)) (n (_ BitVec 32)) (m (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( let ( ( a!1 ( or ( and ( bvsge x n ) ( = x! x ) ( = m! m ) ( = n! n ) ) ( and ( bvslt x n ) ( = x! ( bvadd x #x00000001 ) ) ( = m! x ) ( = n! n ) ) ( and ( bvslt x n ) ( = x! ( bvadd x #x00000001 ) ) ( = m! m ) ( = n! n ) ) ) ) ) ( and a!1 ) ))
(define-fun post_fun ((n (_ BitVec 32)) (m (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( or ( and ( bvsle #x00000000 m ) ( not ( bvsle n m ) ) ) ( bvsle n #x00000000 ) ( not ( bvsle n x ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

