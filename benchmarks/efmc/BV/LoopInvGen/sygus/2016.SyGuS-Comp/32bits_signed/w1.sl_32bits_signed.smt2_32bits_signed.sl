(set-logic BV)

(synth-inv inv_fun ((n (_ BitVec 32))(x (_ BitVec 32))))

(define-fun pre_fun ((n (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( and ( = x #x00000000 ) ( bvsge n #x00000000 ) ))
(define-fun trans_fun ((n! (_ BitVec 32)) (x! (_ BitVec 32)) (n (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( and ( = n! n ) ( bvslt x n ) ( = x! ( bvadd x #x00000001 ) ) ))
(define-fun post_fun ((n (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( or ( not ( bvsle n x ) ) ( = x n ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

