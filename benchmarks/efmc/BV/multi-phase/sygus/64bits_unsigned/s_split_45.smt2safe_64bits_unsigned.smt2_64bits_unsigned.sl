(set-logic BV)

(synth-inv inv_fun ((z (_ BitVec 64))(x (_ BitVec 64))(y (_ BitVec 64))))

(define-fun pre_fun ((z (_ BitVec 64)) (x (_ BitVec 64)) (y (_ BitVec 64))) Bool
       ( = x #x0000000000000000 ))
(define-fun trans_fun ((z! (_ BitVec 64)) (y! (_ BitVec 64)) (x! (_ BitVec 64)) (z (_ BitVec 64)) (x (_ BitVec 64)) (y (_ BitVec 64))) Bool
       ( let ( ( a!1 ( = y! ( ite ( bvuge x #x00000000000bae70 ) ( ite ( bvuge x #x00000000000d3510 ) y ( bvadd y #x0000000000000001 ) ) #x0000000000000000 ) ) ) ( a!2 ( = z! ( ite ( bvuge x #x00000000000a1eda ) ( ite ( bvuge x #x00000000000ba57a ) z ( bvadd z #x0000000000000001 ) ) #x0000000000000000 ) ) ) ) ( and ( = x! ( bvadd #x0000000000000001 x ) ) a!1 a!2 ) ))
(define-fun post_fun ((z (_ BitVec 64)) (x (_ BitVec 64)) (y (_ BitVec 64))) Bool
       ( not ( and ( bvule #x00000000000ebbb0 x ) ( not ( = y z ) ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

