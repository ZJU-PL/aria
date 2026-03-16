(set-logic BV)

(synth-inv inv_fun ((v3 (_ BitVec 64))(v2 (_ BitVec 64))(v1 (_ BitVec 64))(n (_ BitVec 64))(x (_ BitVec 64))))

(define-fun pre_fun ((v3 (_ BitVec 64)) (v2 (_ BitVec 64)) (v1 (_ BitVec 64)) (n (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( = x n ))
(define-fun trans_fun ((v3! (_ BitVec 64)) (v2! (_ BitVec 64)) (v1! (_ BitVec 64)) (n! (_ BitVec 64)) (x! (_ BitVec 64)) (v3 (_ BitVec 64)) (v2 (_ BitVec 64)) (v1 (_ BitVec 64)) (n (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( and ( bvsgt x #x0000000000000001 ) ( = x! ( bvsub x #x0000000000000001 ) ) ( = n! n ) ))
(define-fun post_fun ((v3 (_ BitVec 64)) (v2 (_ BitVec 64)) (v1 (_ BitVec 64)) (n (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( not ( and ( bvsle x #x0000000000000001 ) ( not ( = x #x0000000000000001 ) ) ( bvsle #x0000000000000000 n ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

