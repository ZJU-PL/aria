(set-logic BV)

(synth-inv inv_fun ((z (_ BitVec 64))(x (_ BitVec 64))))

(define-fun pre_fun ((z (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( and ( = x #x00000000000003e8 ) ( = z #x0000000000000064 ) ))
(define-fun trans_fun ((z! (_ BitVec 64)) (x! (_ BitVec 64)) (z (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( let ( ( a!1 ( = x! ( ite ( bvslt ( bvsdiv x #x000000000000000a ) z ) ( bvadd x #x0000000000000001 ) ( bvsub x #x0000000000000001 ) ) ) ) ( a!2 ( = z! ( ite ( bvslt ( bvsdiv x #x000000000000000a ) z ) ( bvsub z #x0000000000000001 ) ( bvadd z #x0000000000000001 ) ) ) ) ) ( and a!1 a!2 ) ))
(define-fun post_fun ((z (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( not ( bvsle x z ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

