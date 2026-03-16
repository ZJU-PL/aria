(set-logic BV)

(synth-inv inv_fun ((n (_ BitVec 32))(x (_ BitVec 32))))

(define-fun pre_fun ((n (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( = x n ))
(define-fun trans_fun ((n! (_ BitVec 32)) (x! (_ BitVec 32)) (n (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( and ( bvsgt x #x00000001 ) ( = x! ( bvsub x #x00000001 ) ) ( = n! n ) ))
(define-fun post_fun ((n (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( not ( and ( bvsle x #x00000001 ) ( not ( = x #x00000001 ) ) ( bvsle #x00000000 n ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

