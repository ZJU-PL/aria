(set-logic BV)

(synth-inv inv_fun ((k (_ BitVec 32))(i (_ BitVec 32))))

(define-fun pre_fun ((k (_ BitVec 32)) (i (_ BitVec 32))) Bool
       ( and ( = i #x00000000 ) ( bvsle #x00000000 k ) ( bvsle k #x0000000a ) ))
(define-fun trans_fun ((k! (_ BitVec 32)) (i! (_ BitVec 32)) (k (_ BitVec 32)) (i (_ BitVec 32))) Bool
       ( and ( bvslt i ( bvmul #x000f4240 k ) ) ( = i! ( bvadd i k ) ) ( = k! k ) ))
(define-fun post_fun ((k (_ BitVec 32)) (i (_ BitVec 32))) Bool
       ( or ( not ( bvsle ( bvmul #x000f4240 k ) i ) ) ( = i ( bvmul #x000f4240 k ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

