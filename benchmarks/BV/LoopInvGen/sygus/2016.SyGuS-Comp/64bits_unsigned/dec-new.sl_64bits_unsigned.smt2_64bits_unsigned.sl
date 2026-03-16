(set-logic BV)

(synth-inv inv_fun ((x (_ BitVec 64))))

(define-fun pre_fun ((x (_ BitVec 64))) Bool
       ( = x #x0000000000002710 ))
(define-fun trans_fun ((x! (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( and ( bvugt x #x0000000000000000 ) ( = x! ( bvsub x #x0000000000000001 ) ) ))
(define-fun post_fun ((x (_ BitVec 64))) Bool
       ( t r u e ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

