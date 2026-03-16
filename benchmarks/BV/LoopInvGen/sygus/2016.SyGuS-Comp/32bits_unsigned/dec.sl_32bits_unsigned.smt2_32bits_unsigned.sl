(set-logic BV)

(synth-inv inv_fun ((x (_ BitVec 32))))

(define-fun pre_fun ((x (_ BitVec 32))) Bool
       ( = x #x00000064 ))
(define-fun trans_fun ((x! (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( and ( bvugt x #x00000000 ) ( = x! ( bvsub x #x00000001 ) ) ))
(define-fun post_fun ((x (_ BitVec 32))) Bool
       ( t r u e ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

