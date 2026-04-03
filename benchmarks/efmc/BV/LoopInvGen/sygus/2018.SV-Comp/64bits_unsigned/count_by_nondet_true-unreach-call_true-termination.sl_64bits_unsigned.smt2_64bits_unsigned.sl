(set-logic BV)

(synth-inv inv_fun ((j (_ BitVec 64))(k (_ BitVec 64))(i (_ BitVec 64))))

(define-fun pre_fun ((j (_ BitVec 64)) (k (_ BitVec 64)) (i (_ BitVec 64))) Bool
       ( and ( = i #x0000000000000000 ) ( = k #x0000000000000000 ) ))
(define-fun trans_fun ((j! (_ BitVec 64)) (k! (_ BitVec 64)) (i! (_ BitVec 64)) (j (_ BitVec 64)) (k (_ BitVec 64)) (i (_ BitVec 64))) Bool
       ( and ( bvult i #x00000000000f4240 ) ( bvule #x0000000000000001 j! ) ( bvult j! #x00000000000f4240 ) ( = i! ( bvadd i j! ) ) ( = k! ( bvadd k #x0000000000000001 ) ) ))
(define-fun post_fun ((j (_ BitVec 64)) (k (_ BitVec 64)) (i (_ BitVec 64))) Bool
       ( or ( not ( bvule #x00000000000f4240 i ) ) ( and ( = ( ( _ extract (_ bv63 64) (_ bv20 64) ) k ) #x00000000000 ) ( bvule ( ( _ extract (_ bv19 64) (_ bv0 64) ) k ) #xf4240 ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

