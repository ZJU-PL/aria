(set-logic BV)

(synth-inv inv_fun ((k (_ BitVec 64))(i (_ BitVec 64))))

(define-fun pre_fun ((k (_ BitVec 64)) (i (_ BitVec 64))) Bool
       ( and ( = i #x0000000000000000 ) ( bvule #x0000000000000000 k ) ( bvule k #x000000000000000a ) ))
(define-fun trans_fun ((k! (_ BitVec 64)) (i! (_ BitVec 64)) (k (_ BitVec 64)) (i (_ BitVec 64))) Bool
       ( and ( bvult i ( bvmul #x00000000000f4240 k ) ) ( = i! ( bvadd i k ) ) ( = k! k ) ))
(define-fun post_fun ((k (_ BitVec 64)) (i (_ BitVec 64))) Bool
       ( or ( not ( bvule ( bvmul #x00000000000f4240 k ) i ) ) ( = i ( bvmul #x00000000000f4240 k ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

