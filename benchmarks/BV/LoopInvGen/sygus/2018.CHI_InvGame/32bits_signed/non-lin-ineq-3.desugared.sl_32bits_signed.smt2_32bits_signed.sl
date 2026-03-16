(set-logic BV)

(synth-inv inv_fun ((k (_ BitVec 32))(i (_ BitVec 32))))

(define-fun pre_fun ((k (_ BitVec 32)) (i (_ BitVec 32))) Bool
       ( and ( = i #x00000000 ) ( = k #x00000000 ) ))
(define-fun trans_fun ((k! (_ BitVec 32)) (i! (_ BitVec 32)) (k (_ BitVec 32)) (i (_ BitVec 32))) Bool
       ( and ( bvslt i #x000003e8 ) ( = i! ( bvadd i #x00000001 ) ) ( = k! ( bvadd k ( bvmul i! i! ) ) ) ))
(define-fun post_fun ((k (_ BitVec 32)) (i (_ BitVec 32))) Bool
       ( or ( not ( bvsle #x000003e8 i ) ) ( bvsle #x000f4240 k ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

