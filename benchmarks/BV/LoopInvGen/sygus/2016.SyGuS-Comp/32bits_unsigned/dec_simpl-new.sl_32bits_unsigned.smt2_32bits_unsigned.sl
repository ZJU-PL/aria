(set-logic BV)

(synth-inv inv_fun ((n (_ BitVec 32))(x (_ BitVec 32))))

(define-fun pre_fun ((n (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( = x n ))
(define-fun trans_fun ((n! (_ BitVec 32)) (x! (_ BitVec 32)) (n (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( and ( bvugt x #x00000001 ) ( = x! ( bvsub x #x00000001 ) ) ( = n! n ) ))
(define-fun post_fun ((n (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( not ( and ( = ( ( _ extract (_ bv31 32) (_ bv1 32) ) x ) #b0000000000000000000000000000000 ) ( not ( = x #x00000001 ) ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

