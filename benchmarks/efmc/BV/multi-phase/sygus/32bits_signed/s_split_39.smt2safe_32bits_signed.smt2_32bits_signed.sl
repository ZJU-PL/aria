(set-logic BV)

(synth-inv inv_fun ((z (_ BitVec 32))(x (_ BitVec 32))))

(define-fun pre_fun ((z (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( and ( = x #x00000000 ) ( = z #x00000000 ) ))
(define-fun trans_fun ((z! (_ BitVec 32)) (x! (_ BitVec 32)) (z (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( let ( ( a!1 ( = x! ( ite ( bvslt ( bvmul x #x00000005 ) z ) ( bvadd x #x00000001 ) ( bvsdiv x #x0000000a ) ) ) ) ( a!2 ( = z! ( ite ( bvslt ( bvmul x #x00000005 ) z ) z ( bvadd #x00000001 z ) ) ) ) ) ( and a!1 a!2 ) ))
(define-fun post_fun ((z (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( not ( and ( not ( bvsle z #x00000032 ) ) ( bvsle z x ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

