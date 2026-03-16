(set-logic BV)

(synth-inv inv_fun ((z (_ BitVec 64))(x (_ BitVec 64))(y (_ BitVec 64))))

(define-fun pre_fun ((z (_ BitVec 64)) (x (_ BitVec 64)) (y (_ BitVec 64))) Bool
       ( and ( = x #x0000000000000000 ) ( = y #x0000000000000000 ) ( = z #x0000000000000000 ) ))
(define-fun trans_fun ((z! (_ BitVec 64)) (y! (_ BitVec 64)) (x! (_ BitVec 64)) (z (_ BitVec 64)) (x (_ BitVec 64)) (y (_ BitVec 64))) Bool
       ( and ( = x! ( bvadd x #x0000000000000001 ) ) ( = y! ( bvadd y x ) ) ( = z! ( ite ( bvsgt y x ) ( bvadd z #x0000000000000001 ) z ) ) ))
(define-fun post_fun ((z (_ BitVec 64)) (x (_ BitVec 64)) (y (_ BitVec 64))) Bool
       ( not ( and ( bvsle #x000000000034adcd x ) ( not ( bvsle z #x0000000000000000 ) ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

