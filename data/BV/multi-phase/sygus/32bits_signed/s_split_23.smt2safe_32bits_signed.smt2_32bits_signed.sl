(set-logic BV)

(synth-inv inv_fun ((z (_ BitVec 32))(x (_ BitVec 32))))

(define-fun pre_fun ((z (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( and ( = x #x000003e8 ) ( = z #x00000064 ) ))
(define-fun trans_fun ((z! (_ BitVec 32)) (x! (_ BitVec 32)) (z (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( let ( ( a!1 ( = x! ( ite ( bvslt ( bvsdiv x #x0000000a ) z ) ( bvadd x #x00000001 ) ( bvsub x #x00000001 ) ) ) ) ( a!2 ( = z! ( ite ( bvslt ( bvsdiv x #x0000000a ) z ) ( bvsub z #x00000001 ) ( bvadd z #x00000001 ) ) ) ) ) ( and a!1 a!2 ) ))
(define-fun post_fun ((z (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( not ( bvsle x z ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

