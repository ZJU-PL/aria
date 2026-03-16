(set-logic BV)

(synth-inv inv_fun ((z (_ BitVec 32))(x (_ BitVec 32))(y (_ BitVec 32))))

(define-fun pre_fun ((z (_ BitVec 32)) (x (_ BitVec 32)) (y (_ BitVec 32))) Bool
       ( = x #x00000000 ))
(define-fun trans_fun ((z! (_ BitVec 32)) (y! (_ BitVec 32)) (x! (_ BitVec 32)) (z (_ BitVec 32)) (x (_ BitVec 32)) (y (_ BitVec 32))) Bool
       ( let ( ( a!1 ( = y! ( ite ( bvuge x #x000bae70 ) ( ite ( bvuge x #x000d3510 ) y ( bvadd y #x00000001 ) ) #x00000000 ) ) ) ( a!2 ( = z! ( ite ( bvuge x #x000a1eda ) ( ite ( bvuge x #x000ba57a ) z ( bvadd z #x00000001 ) ) #x00000000 ) ) ) ) ( and ( = x! ( bvadd #x00000001 x ) ) a!1 a!2 ) ))
(define-fun post_fun ((z (_ BitVec 32)) (x (_ BitVec 32)) (y (_ BitVec 32))) Bool
       ( not ( and ( bvule #x000ebbb0 x ) ( not ( = y z ) ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

