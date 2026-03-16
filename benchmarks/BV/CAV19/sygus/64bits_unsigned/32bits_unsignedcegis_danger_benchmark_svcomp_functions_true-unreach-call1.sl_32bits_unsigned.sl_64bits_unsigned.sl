(set-logic BV)

(synth-inv inv_fun ((x (_ BitVec 64))))

(define-fun pre_fun ((x (_ BitVec 64))) Bool
       ( = x #x00000000 ))
(define-fun trans_fun ((x (_ BitVec 64)) (x! (_ BitVec 64))) Bool
       ( and ( bvult x #x0fffffff ) ( = x! ( bvadd #x00000002 x ) ) ))
(define-fun post_fun ((x (_ BitVec 64))) Bool
       ( or ( = #x00000000 ( bvurem x #x00000002 ) ) ( bvult x #x0fffffff ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

