(set-logic BV)

(synth-inv inv_fun ((z (_ BitVec 32))(y (_ BitVec 32))(x (_ BitVec 32))))

(define-fun pre_fun ((z (_ BitVec 32)) (y (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( and ( = x #x00000000 ) ( = y z ) ( = z #x00000000 ) ))
(define-fun trans_fun ((z! (_ BitVec 32)) (y! (_ BitVec 32)) (x! (_ BitVec 32)) (z (_ BitVec 32)) (y (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( and ( = x! ( bvadd x #x00000001 ) ) ( = y! ( ite ( bvsge x #x000006e5 ) ( bvadd y #x00000002 ) ( bvadd y #x00000001 ) ) ) ( = z! ( ite ( bvsge y #x00001685 ) ( bvadd z #x00000003 ) ( bvadd z #x00000002 ) ) ) ))
(define-fun post_fun ((z (_ BitVec 32)) (y (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( not ( and ( not ( bvsle x #x000044f2 ) ) ( not ( bvsle z #x00006c02 ) ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

