(set-logic BV)

(synth-inv inv_fun ((x (_ BitVec 64))))

(define-fun pre_fun ((x (_ BitVec 64))) Bool
       ( = x #x0000000000000000 ))
(define-fun trans_fun ((x! (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( and ( = x! #x0000000000000000 ) ))
(define-fun post_fun ((x (_ BitVec 64))) Bool
       ( = x #x0000000000000000 ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

