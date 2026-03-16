(set-logic BV)

(synth-inv inv_fun ((z (_ BitVec 64))(y (_ BitVec 64))(x (_ BitVec 64))))

(define-fun pre_fun ((z (_ BitVec 64)) (y (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( and ( = x #x0000000000000000 ) ( = y z ) ( = z #x0000000000000000 ) ))
(define-fun trans_fun ((z! (_ BitVec 64)) (y! (_ BitVec 64)) (x! (_ BitVec 64)) (z (_ BitVec 64)) (y (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( and ( = x! ( bvadd x #x0000000000000001 ) ) ( = y! ( ite ( bvsge x #x00000000000006e5 ) ( bvadd y #x0000000000000002 ) ( bvadd y #x0000000000000001 ) ) ) ( = z! ( ite ( bvsge y #x0000000000001685 ) ( bvadd z #x0000000000000003 ) ( bvadd z #x0000000000000002 ) ) ) ))
(define-fun post_fun ((z (_ BitVec 64)) (y (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( not ( and ( not ( bvsle x #x00000000000044f2 ) ) ( bvsle z #x0000000000006c02 ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

