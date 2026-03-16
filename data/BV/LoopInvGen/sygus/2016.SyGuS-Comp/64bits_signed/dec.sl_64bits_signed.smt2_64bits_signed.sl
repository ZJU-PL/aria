(set-logic BV)

(synth-inv inv_fun ((x (_ BitVec 64))))

(define-fun pre_fun ((x (_ BitVec 64))) Bool
       ( = x #x0000000000000064 ))
(define-fun trans_fun ((x! (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( and ( bvsgt x #x0000000000000000 ) ( = x! ( bvsub x #x0000000000000001 ) ) ))
(define-fun post_fun ((x (_ BitVec 64))) Bool
       ( not ( and ( bvsle x #x0000000000000000 ) ( not ( = x #x0000000000000000 ) ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

