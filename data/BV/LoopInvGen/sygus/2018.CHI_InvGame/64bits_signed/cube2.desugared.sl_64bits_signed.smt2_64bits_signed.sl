(set-logic BV)

(synth-inv inv_fun ((z (_ BitVec 64))(y (_ BitVec 64))(x (_ BitVec 64))(n (_ BitVec 64))))

(define-fun pre_fun ((z (_ BitVec 64)) (y (_ BitVec 64)) (x (_ BitVec 64)) (n (_ BitVec 64))) Bool
       ( and ( = n #x0000000000000000 ) ( = x #x0000000000000000 ) ( = y #x0000000000000001 ) ( = z #x0000000000000006 ) ))
(define-fun trans_fun ((z! (_ BitVec 64)) (y! (_ BitVec 64)) (x! (_ BitVec 64)) (n! (_ BitVec 64)) (z (_ BitVec 64)) (y (_ BitVec 64)) (x (_ BitVec 64)) (n (_ BitVec 64))) Bool
       ( and ( bvslt n #x0000000000000064 ) ( = n! ( bvadd n #x0000000000000001 ) ) ( = x! ( bvadd x y ) ) ( = y! ( bvadd y z ) ) ( = z! ( bvadd z #x0000000000000006 ) ) ))
(define-fun post_fun ((z (_ BitVec 64)) (y (_ BitVec 64)) (x (_ BitVec 64)) (n (_ BitVec 64))) Bool
       ( or ( not ( bvsle #x0000000000000064 n ) ) ( = x ( bvmul n n n ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

