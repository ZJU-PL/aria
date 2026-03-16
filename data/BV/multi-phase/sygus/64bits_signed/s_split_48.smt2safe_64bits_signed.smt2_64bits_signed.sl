(set-logic BV)

(synth-inv inv_fun ((y (_ BitVec 64))(x (_ BitVec 64))))

(define-fun pre_fun ((y (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( and ( = x #x0000000000000000 ) ( = y #x0000000000000000 ) ))
(define-fun trans_fun ((y! (_ BitVec 64)) (x! (_ BitVec 64)) (y (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( let ( ( a!1 ( = y! ( ite ( bvslt x #x0000000000001388 ) ( ite ( bvsge x #x0000000000000fa0 ) ( bvadd y #x0000000000000004 ) ( bvadd y #x0000000000000001 ) ) ( ite ( bvsge x #x0000000000001770 ) ( bvsub y #x0000000000000001 ) ( bvsub y #x0000000000000004 ) ) ) ) ) ) ( and ( = x! ( bvadd x #x0000000000000001 ) ) a!1 ) ))
(define-fun post_fun ((y (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( not ( and ( = x #x0000000000002710 ) ( not ( = y #x0000000000000000 ) ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

