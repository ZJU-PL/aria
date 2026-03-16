(set-logic BV)

(synth-inv inv_fun ((x (_ BitVec 64))))

(define-fun pre_fun ((x (_ BitVec 64))) Bool
       ( = x #x0000000a ))
(define-fun trans_fun ((x (_ BitVec 64)) (x! (_ BitVec 64))) Bool
       ( and ( bvuge x #x0000000a ) ( = x! ( bvadd x #x00000002 ) ) ))
(define-fun post_fun ((x (_ BitVec 64))) Bool
       ( or ( = #x00000000 ( bvurem x #x00000002 ) ) ( bvuge x #x0000000a ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

