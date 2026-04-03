(set-logic BV)

(synth-inv inv_fun ((k (_ BitVec 64))(j (_ BitVec 64))(i (_ BitVec 64))))

(define-fun pre_fun ((k (_ BitVec 64)) (j (_ BitVec 64)) (i (_ BitVec 64))) Bool
       ( and ( = i #x0000000000000001 ) ( = j #x0000000000000001 ) ( bvule #x0000000000000000 k ) ( bvule k #x0000000000000001 ) ))
(define-fun trans_fun ((k! (_ BitVec 64)) (j! (_ BitVec 64)) (i! (_ BitVec 64)) (k (_ BitVec 64)) (j (_ BitVec 64)) (i (_ BitVec 64))) Bool
       ( and ( bvult i #x00000000000f4240 ) ( = i! ( bvadd i #x0000000000000001 ) ) ( = j! ( bvadd j k ) ) ( = k! ( bvsub k #x0000000000000001 ) ) ))
(define-fun post_fun ((k (_ BitVec 64)) (j (_ BitVec 64)) (i (_ BitVec 64))) Bool
       ( let ( ( a!1 ( and ( bvule #x0000000000000001 ( bvadd i k ) ) ( = ( ( _ extract (_ bv63 64) (_ bv2 64) ) ( bvadd i k ) ) #b00000000000000000000000000000000000000000000000000000000000000 ) ( bvule ( bvadd ( ( _ extract (_ bv1 64) (_ bv0 64) ) i ) ( ( _ extract (_ bv1 64) (_ bv0 64) ) k ) ) #b10 ) ( bvule #x0000000000000001 i ) ) ) ) ( or ( bvule #x00000000000f4240 i ) a!1 ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

