(set-logic BV)

(synth-inv inv_fun ((x (_ BitVec 32))))

(define-fun pre_fun ((x (_ BitVec 32))) Bool
       ( = x #x0ffffff0 ))
(define-fun trans_fun ((x (_ BitVec 32)) (x! (_ BitVec 32))) Bool
       ( and ( bvugt x #x00000000 ) ( = x! ( bvsub x #x00000002 ) ) ))
(define-fun post_fun ((x (_ BitVec 32))) Bool
       ( or ( = #x00000000 ( bvurem x #x00000002 ) ) ( bvugt x #x00000000 ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

