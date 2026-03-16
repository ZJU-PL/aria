(set-logic BV)

(synth-inv inv_fun ((z (_ BitVec 64))(x (_ BitVec 64))(y (_ BitVec 64))))

(define-fun pre_fun ((z (_ BitVec 64)) (x (_ BitVec 64)) (y (_ BitVec 64))) Bool
       ( and ( = x #x0000000000000000 ) ( bvsge y #x0000000000000064 ) ( = z #x0000000000000000 ) ))
(define-fun trans_fun ((z! (_ BitVec 64)) (y! (_ BitVec 64)) (x! (_ BitVec 64)) (z (_ BitVec 64)) (x (_ BitVec 64)) (y (_ BitVec 64))) Bool
       ( let ( ( a!1 ( = z! ( ite ( bvsle y ( bvsdiv x #x0000000000000032 ) ) ( bvadd z #x0000000000000001 ) z ) ) ) ) ( and ( = x! ( bvadd #x0000000000000001 x ) ) ( = y! ( bvsub y #x0000000000000001 ) ) a!1 ) ))
(define-fun post_fun ((z (_ BitVec 64)) (x (_ BitVec 64)) (y (_ BitVec 64))) Bool
       ( not ( and ( = y #x0000000000000000 ) ( bvsle z #x0000000000000000 ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

