(set-logic BV)

(synth-inv inv_fun ((i (_ BitVec 64))))

(define-fun pre_fun ((i (_ BitVec 64))) Bool
       ( = i #x0000000000000000 ))
(define-fun trans_fun ((i! (_ BitVec 64)) (i (_ BitVec 64))) Bool
       ( and ( not ( = i #x00000000000f4240 ) ) ( = i! ( bvadd i #x0000000000000001 ) ) ))
(define-fun post_fun ((i (_ BitVec 64))) Bool
       ( or ( not ( = i #x00000000000f4240 ) ) ( bvsle i #x00000000000f4240 ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

