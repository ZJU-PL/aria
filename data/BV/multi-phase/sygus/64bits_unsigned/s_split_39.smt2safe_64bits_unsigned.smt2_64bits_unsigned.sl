(set-logic BV)

(synth-inv inv_fun ((z (_ BitVec 64))(x (_ BitVec 64))))

(define-fun pre_fun ((z (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( and ( = x #x0000000000000000 ) ( = z #x0000000000000000 ) ))
(define-fun trans_fun ((z! (_ BitVec 64)) (x! (_ BitVec 64)) (z (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( let ( ( a!1 ( = x! ( ite ( bvult ( bvmul x #x0000000000000005 ) z ) ( bvadd x #x0000000000000001 ) ( bvudiv x #x000000000000000a ) ) ) ) ( a!2 ( = z! ( ite ( bvult ( bvmul x #x0000000000000005 ) z ) z ( bvadd #x0000000000000001 z ) ) ) ) ) ( and a!1 a!2 ) ))
(define-fun post_fun ((z (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( let ( ( a!1 ( not ( and ( = ( ( _ extract (_ bv63 64) (_ bv6 64) ) z ) #b0000000000000000000000000000000000000000000000000000000000 ) ( bvule ( ( _ extract (_ bv5 64) (_ bv0 64) ) z ) #b110010 ) ) ) ) ) ( not ( and a!1 ( bvule z x ) ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

