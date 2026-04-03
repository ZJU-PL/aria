(set-logic BV)

(synth-inv inv_fun ((z (_ BitVec 32))(y (_ BitVec 32))(x (_ BitVec 32))))

(define-fun pre_fun ((z (_ BitVec 32)) (y (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( and ( = x #x00000000 ) ( = y #x000003e8 ) ( = z #x000007d0 ) ))
(define-fun trans_fun ((z! (_ BitVec 32)) (y! (_ BitVec 32)) (x! (_ BitVec 32)) (z (_ BitVec 32)) (y (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( and ( = x! ( bvadd #x00000001 x ) ) ( = y! ( ite ( bvsge x #x000003e8 ) ( bvadd y #x00000001 ) y ) ) ( = z! ( ite ( bvsge y #x000007d0 ) ( bvadd z #x00000001 ) z ) ) ))
(define-fun post_fun ((z (_ BitVec 32)) (y (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( not ( and ( bvsle #x000007d0 y ) ( = x z ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

