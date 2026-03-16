(set-logic BV)

(synth-inv inv_fun ((x (_ BitVec 64))))

(define-fun pre_fun ((x (_ BitVec 64))) Bool
       ( = x #x0000000000000000 ))
(define-fun trans_fun ((x! (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( and ( bvult x #x0000000000000064 ) ( = x! ( bvadd x #x0000000000000001 ) ) ))
(define-fun post_fun ((x (_ BitVec 64))) Bool
       ( or ( not ( bvule #x0000000000000064 x ) ) ( = x #x0000000000000064 ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

