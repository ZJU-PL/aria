(set-logic BV)

(synth-inv inv_fun ((y (_ BitVec 64))(x (_ BitVec 64))))

(define-fun pre_fun ((y (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( and ( = x #x0000000000000000 ) ( = y #x0000000000000000 ) ))
(define-fun trans_fun ((y! (_ BitVec 64)) (x! (_ BitVec 64)) (y (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( let ( ( a!1 ( = y! ( ite ( bvsge x #x0000000002faf080 ) ( ite ( bvsge x #x0000000005f5e100 ) y ( bvadd y #x0000000000000001 ) ) #x0000000000000000 ) ) ) ) ( and ( = x! ( bvadd #x0000000000000001 x ) ) a!1 ) ))
(define-fun post_fun ((y (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( not ( and ( bvsle #x0000000005f5e100 x ) ( not ( = y #x0000000002faf080 ) ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

