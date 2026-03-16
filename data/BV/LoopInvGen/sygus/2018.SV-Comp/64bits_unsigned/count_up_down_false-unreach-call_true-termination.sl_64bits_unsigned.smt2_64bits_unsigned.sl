(set-logic BV)

(synth-inv inv_fun ((y (_ BitVec 64))(x (_ BitVec 64))(n (_ BitVec 64))))

(define-fun pre_fun ((y (_ BitVec 64)) (x (_ BitVec 64)) (n (_ BitVec 64))) Bool
       ( and ( = y #x0000000000000000 ) ( = x n ) ))
(define-fun trans_fun ((y! (_ BitVec 64)) (x! (_ BitVec 64)) (n! (_ BitVec 64)) (y (_ BitVec 64)) (x (_ BitVec 64)) (n (_ BitVec 64))) Bool
       ( and ( bvugt x #x0000000000000000 ) ( = x! ( bvsub x #x0000000000000001 ) ) ( = y! ( bvadd y #x0000000000000001 ) ) ( = n! n ) ))
(define-fun post_fun ((y (_ BitVec 64)) (x (_ BitVec 64)) (n (_ BitVec 64))) Bool
       ( or ( not ( = x #x0000000000000000 ) ) ( = y n ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

