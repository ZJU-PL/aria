(set-logic BV)

(synth-inv inv_fun ((v3 (_ BitVec 32))(v2 (_ BitVec 32))(v1 (_ BitVec 32))(n (_ BitVec 32))(x (_ BitVec 32))))

(define-fun pre_fun ((v3 (_ BitVec 32)) (v2 (_ BitVec 32)) (v1 (_ BitVec 32)) (n (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( = x #x00000000 ))
(define-fun trans_fun ((v3! (_ BitVec 32)) (v2! (_ BitVec 32)) (v1! (_ BitVec 32)) (n! (_ BitVec 32)) (x! (_ BitVec 32)) (v3 (_ BitVec 32)) (v2 (_ BitVec 32)) (v1 (_ BitVec 32)) (n (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( and ( = n! n ) ( bvslt x n ) ( = x! ( bvadd x #x00000001 ) ) ))
(define-fun post_fun ((v3 (_ BitVec 32)) (v2 (_ BitVec 32)) (v1 (_ BitVec 32)) (n (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( or ( not ( bvsle #x00000000 n ) ) ( = x n ) ( not ( bvsle n x ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

