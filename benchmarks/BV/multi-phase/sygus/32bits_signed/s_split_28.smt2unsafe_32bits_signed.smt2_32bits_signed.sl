(set-logic BV)

(synth-inv inv_fun ((z (_ BitVec 32))(x (_ BitVec 32))(y (_ BitVec 32))))

(define-fun pre_fun ((z (_ BitVec 32)) (x (_ BitVec 32)) (y (_ BitVec 32))) Bool
       ( and ( = x #x00000000 ) ( bvsge y #x00000064 ) ( = z #x00000000 ) ))
(define-fun trans_fun ((z! (_ BitVec 32)) (y! (_ BitVec 32)) (x! (_ BitVec 32)) (z (_ BitVec 32)) (x (_ BitVec 32)) (y (_ BitVec 32))) Bool
       ( let ( ( a!1 ( = z! ( ite ( bvsle y ( bvsdiv x #x00000032 ) ) ( bvadd z #x00000001 ) z ) ) ) ) ( and ( = x! ( bvadd #x00000001 x ) ) ( = y! ( bvsub y #x00000001 ) ) a!1 ) ))
(define-fun post_fun ((z (_ BitVec 32)) (x (_ BitVec 32)) (y (_ BitVec 32))) Bool
       ( not ( and ( = y #x00000000 ) ( not ( bvsle z #x00000000 ) ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

