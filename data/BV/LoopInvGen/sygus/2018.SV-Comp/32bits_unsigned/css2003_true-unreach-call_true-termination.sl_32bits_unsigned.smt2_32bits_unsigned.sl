(set-logic BV)

(synth-inv inv_fun ((k (_ BitVec 32))(j (_ BitVec 32))(i (_ BitVec 32))))

(define-fun pre_fun ((k (_ BitVec 32)) (j (_ BitVec 32)) (i (_ BitVec 32))) Bool
       ( and ( = i #x00000001 ) ( = j #x00000001 ) ( bvule #x00000000 k ) ( bvule k #x00000001 ) ))
(define-fun trans_fun ((k! (_ BitVec 32)) (j! (_ BitVec 32)) (i! (_ BitVec 32)) (k (_ BitVec 32)) (j (_ BitVec 32)) (i (_ BitVec 32))) Bool
       ( and ( bvult i #x000f4240 ) ( = i! ( bvadd i #x00000001 ) ) ( = j! ( bvadd j k ) ) ( = k! ( bvsub k #x00000001 ) ) ))
(define-fun post_fun ((k (_ BitVec 32)) (j (_ BitVec 32)) (i (_ BitVec 32))) Bool
       ( let ( ( a!1 ( and ( bvule #x00000001 ( bvadd i k ) ) ( = ( ( _ extract (_ bv31 32) (_ bv2 32) ) ( bvadd i k ) ) #b000000000000000000000000000000 ) ( bvule ( bvadd ( ( _ extract (_ bv1 32) (_ bv0 32) ) i ) ( ( _ extract (_ bv1 32) (_ bv0 32) ) k ) ) #b10 ) ( bvule #x00000001 i ) ) ) ) ( or ( bvule #x000f4240 i ) a!1 ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

