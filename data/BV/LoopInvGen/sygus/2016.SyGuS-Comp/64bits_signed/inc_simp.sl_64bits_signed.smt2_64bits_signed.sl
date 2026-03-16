(set-logic BV)

(synth-inv inv_fun ((n (_ BitVec 64))(x (_ BitVec 64))))

(define-fun pre_fun ((n (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( = x #x0000000000000000 ))
(define-fun trans_fun ((n! (_ BitVec 64)) (x! (_ BitVec 64)) (n (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( and ( = n! n ) ( bvslt x n ) ( = x! ( bvadd x #x0000000000000001 ) ) ))
(define-fun post_fun ((n (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( or ( not ( bvsle n x ) ) ( = x n ) ( not ( bvsle #x0000000000000000 n ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

