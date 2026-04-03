(set-logic BV)

(synth-inv inv_fun ((y (_ BitVec 32))(x (_ BitVec 32))))

(define-fun pre_fun ((y (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( and ( = x #x00000000 ) ( = y #x00000000 ) ))
(define-fun trans_fun ((y! (_ BitVec 32)) (x! (_ BitVec 32)) (y (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( let ( ( a!1 ( = y! ( ite ( bvsge x #x00001d4c ) ( ite ( bvsge x #x000030d4 ) ( bvsub y #x00000002 ) ( bvadd y #x00000001 ) ) ( ite ( bvsge x #x000009c4 ) ( bvadd y #x00000001 ) ( bvsub y #x00000002 ) ) ) ) ) ) ( and ( = x! ( bvadd x #x00000001 ) ) a!1 ) ))
(define-fun post_fun ((y (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( not ( and ( = x #x00003a98 ) ( not ( = y #x00000000 ) ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

