(set-logic BV)

(synth-inv inv_fun ((z (_ BitVec 64))(x (_ BitVec 64))))

(define-fun pre_fun ((z (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( and ( = x #x0000000000000000 ) ( = z #x0000000000000000 ) ))
(define-fun trans_fun ((z! (_ BitVec 64)) (x! (_ BitVec 64)) (z (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( let ( ( a!1 ( = x! ( ite ( bvslt ( bvmul x #x0000000000000005 ) z ) ( bvadd x #x0000000000000001 ) ( bvsdiv x #x000000000000000a ) ) ) ) ( a!2 ( = z! ( ite ( bvslt ( bvmul x #x0000000000000005 ) z ) z ( bvadd #x0000000000000001 z ) ) ) ) ) ( and a!1 a!2 ) ))
(define-fun post_fun ((z (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( not ( and ( not ( bvsle z #x0000000000000032 ) ) ( not ( bvsle z x ) ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

