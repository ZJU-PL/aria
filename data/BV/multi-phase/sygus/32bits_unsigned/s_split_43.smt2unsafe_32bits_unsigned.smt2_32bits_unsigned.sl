(set-logic BV)

(synth-inv inv_fun ((y (_ BitVec 32))(x (_ BitVec 32))))

(define-fun pre_fun ((y (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( and ( = x #x00000000 ) ( = y #x00000000 ) ))
(define-fun trans_fun ((y! (_ BitVec 32)) (x! (_ BitVec 32)) (y (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( let ( ( a!1 ( = y! ( ite ( bvuge x #x02faf080 ) ( ite ( bvuge x #x05f5e100 ) y ( bvadd y #x00000001 ) ) #x00000000 ) ) ) ) ( and ( = x! ( bvadd #x00000001 x ) ) a!1 ) ))
(define-fun post_fun ((y (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( not ( and ( bvule #x05f5e100 x ) ( = y #x02faf080 ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

