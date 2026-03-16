(set-logic BV)

(synth-inv inv_fun ((z (_ BitVec 32))(x (_ BitVec 32))(y (_ BitVec 32))))

(define-fun pre_fun ((z (_ BitVec 32)) (x (_ BitVec 32)) (y (_ BitVec 32))) Bool
       ( and ( = x #x00000000 ) ( = y #x00000000 ) ( = z #x00000000 ) ))
(define-fun trans_fun ((z! (_ BitVec 32)) (y! (_ BitVec 32)) (x! (_ BitVec 32)) (z (_ BitVec 32)) (x (_ BitVec 32)) (y (_ BitVec 32))) Bool
       ( and ( = x! ( bvadd x #x00000001 ) ) ( = y! ( bvadd y x ) ) ( = z! ( ite ( bvsgt y x ) ( bvadd z #x00000001 ) z ) ) ))
(define-fun post_fun ((z (_ BitVec 32)) (x (_ BitVec 32)) (y (_ BitVec 32))) Bool
       ( not ( and ( bvsle #x0034adcd x ) ( not ( bvsle z #x00000000 ) ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

