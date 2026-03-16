(set-logic BV)

(synth-inv inv_fun ((y (_ BitVec 64))(x (_ BitVec 64))))

(define-fun pre_fun ((y (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( and ( = x #x0000000000000000 ) ( = y #x0000000000000000 ) ))
(define-fun trans_fun ((y! (_ BitVec 64)) (x! (_ BitVec 64)) (y (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( let ( ( a!1 ( = y! ( ite ( bvsge x #x0000000000001d4c ) ( ite ( bvsge x #x00000000000030d4 ) ( bvsub y #x0000000000000002 ) ( bvadd y #x0000000000000001 ) ) ( ite ( bvsge x #x00000000000009c4 ) ( bvadd y #x0000000000000001 ) ( bvsub y #x0000000000000002 ) ) ) ) ) ) ( and ( = x! ( bvadd x #x0000000000000001 ) ) a!1 ) ))
(define-fun post_fun ((y (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( not ( and ( = x #x0000000000003a98 ) ( = y #x0000000000000000 ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

