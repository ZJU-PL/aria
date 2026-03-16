(set-logic BV)

(synth-inv inv_fun ((y (_ BitVec 32))(x (_ BitVec 32))(n (_ BitVec 32))))

(define-fun pre_fun ((y (_ BitVec 32)) (x (_ BitVec 32)) (n (_ BitVec 32))) Bool
       ( and ( = y #x00000000 ) ( = x n ) ))
(define-fun trans_fun ((y! (_ BitVec 32)) (x! (_ BitVec 32)) (n! (_ BitVec 32)) (y (_ BitVec 32)) (x (_ BitVec 32)) (n (_ BitVec 32))) Bool
       ( and ( bvsgt x #x00000000 ) ( = x! ( bvsub x #x00000001 ) ) ( = y! ( bvadd y #x00000001 ) ) ( = n! n ) ))
(define-fun post_fun ((y (_ BitVec 32)) (x (_ BitVec 32)) (n (_ BitVec 32))) Bool
       ( or ( not ( bvsle x #x00000000 ) ) ( = y n ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

