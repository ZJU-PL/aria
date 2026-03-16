(set-logic BV)

(synth-inv inv_fun ((x (_ BitVec 64))(y (_ BitVec 64))))

(define-fun pre_fun ((x (_ BitVec 64)) (y (_ BitVec 64))) Bool
       ( and ( = x #x00000001 ) ( = y #x00000000 ) ))
(define-fun trans_fun ((x (_ BitVec 64)) (x! (_ BitVec 64)) (y (_ BitVec 64)) (y! (_ BitVec 64))) Bool
       ( and ( bvult y #x00000400 ) ( = x! #x00000000 ) ( = y! ( bvadd y #x00000001 ) ) ))
(define-fun post_fun ((x (_ BitVec 64)) (y (_ BitVec 64))) Bool
       ( or ( = x #x00000000 ) ( bvult y #x00000400 ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

