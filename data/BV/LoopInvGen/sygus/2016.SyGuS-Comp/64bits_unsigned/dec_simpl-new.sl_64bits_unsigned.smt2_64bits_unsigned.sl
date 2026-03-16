(set-logic BV)

(synth-inv inv_fun ((n (_ BitVec 64))(x (_ BitVec 64))))

(define-fun pre_fun ((n (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( = x n ))
(define-fun trans_fun ((n! (_ BitVec 64)) (x! (_ BitVec 64)) (n (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( and ( bvugt x #x0000000000000001 ) ( = x! ( bvsub x #x0000000000000001 ) ) ( = n! n ) ))
(define-fun post_fun ((n (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( not ( and ( = ( ( _ extract (_ bv63 64) (_ bv1 64) ) x ) #b000000000000000000000000000000000000000000000000000000000000000 ) ( not ( = x #x0000000000000001 ) ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

