(set-logic BV)

(synth-inv inv_fun ((x (_ BitVec 32))))

(define-fun pre_fun ((x (_ BitVec 32))) Bool
       ( = x #x0000000a ))
(define-fun trans_fun ((x (_ BitVec 32)) (x! (_ BitVec 32))) Bool
       ( and ( bvsge x #x0000000a ) ( = x! ( bvadd x #x00000002 ) ) ))
(define-fun post_fun ((x (_ BitVec 32))) Bool
       ( or ( = #x00000000 ( bvurem x #x00000002 ) ) ( bvsge x #x0000000a ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

