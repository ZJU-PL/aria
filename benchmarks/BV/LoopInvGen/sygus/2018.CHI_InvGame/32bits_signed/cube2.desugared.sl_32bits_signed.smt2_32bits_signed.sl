(set-logic BV)

(synth-inv inv_fun ((z (_ BitVec 32))(y (_ BitVec 32))(x (_ BitVec 32))(n (_ BitVec 32))))

(define-fun pre_fun ((z (_ BitVec 32)) (y (_ BitVec 32)) (x (_ BitVec 32)) (n (_ BitVec 32))) Bool
       ( and ( = n #x00000000 ) ( = x #x00000000 ) ( = y #x00000001 ) ( = z #x00000006 ) ))
(define-fun trans_fun ((z! (_ BitVec 32)) (y! (_ BitVec 32)) (x! (_ BitVec 32)) (n! (_ BitVec 32)) (z (_ BitVec 32)) (y (_ BitVec 32)) (x (_ BitVec 32)) (n (_ BitVec 32))) Bool
       ( and ( bvslt n #x00000064 ) ( = n! ( bvadd n #x00000001 ) ) ( = x! ( bvadd x y ) ) ( = y! ( bvadd y z ) ) ( = z! ( bvadd z #x00000006 ) ) ))
(define-fun post_fun ((z (_ BitVec 32)) (y (_ BitVec 32)) (x (_ BitVec 32)) (n (_ BitVec 32))) Bool
       ( or ( not ( bvsle #x00000064 n ) ) ( = x ( bvmul n n n ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

