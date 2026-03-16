(set-logic BV)

(synth-inv inv_fun ((j (_ BitVec 32))(k (_ BitVec 32))(i (_ BitVec 32))))

(define-fun pre_fun ((j (_ BitVec 32)) (k (_ BitVec 32)) (i (_ BitVec 32))) Bool
       ( and ( = i #x00000000 ) ( = k #x00000000 ) ))
(define-fun trans_fun ((j! (_ BitVec 32)) (k! (_ BitVec 32)) (i! (_ BitVec 32)) (j (_ BitVec 32)) (k (_ BitVec 32)) (i (_ BitVec 32))) Bool
       ( and ( bvult i #x000f4240 ) ( bvule #x00000001 j! ) ( bvult j! #x000f4240 ) ( = i! ( bvadd i j! ) ) ( = k! ( bvadd k #x00000001 ) ) ))
(define-fun post_fun ((j (_ BitVec 32)) (k (_ BitVec 32)) (i (_ BitVec 32))) Bool
       ( or ( not ( bvule #x000f4240 i ) ) ( and ( = ( ( _ extract (_ bv31 32) (_ bv20 32) ) k ) #x000 ) ( bvule ( ( _ extract (_ bv19 32) (_ bv0 32) ) k ) #xf4240 ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

