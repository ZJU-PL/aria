(set-logic BV)

(synth-inv inv_fun ((y (_ BitVec 32))(x (_ BitVec 32))))

(define-fun pre_fun ((y (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( and ( = x #x00000000 ) ( = y #x00000000 ) ))
(define-fun trans_fun ((y! (_ BitVec 32)) (x! (_ BitVec 32)) (y (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( let ( ( a!1 ( = y! ( ite ( bvult x #x00001388 ) ( ite ( bvuge x #x00000fa0 ) ( bvadd y #x00000004 ) ( bvadd y #x00000001 ) ) ( ite ( bvuge x #x00001770 ) ( bvsub y #x00000001 ) ( bvsub y #x00000004 ) ) ) ) ) ) ( and ( = x! ( bvadd x #x00000001 ) ) a!1 ) ))
(define-fun post_fun ((y (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( not ( and ( = x #x00002710 ) ( not ( = y #x00000000 ) ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

