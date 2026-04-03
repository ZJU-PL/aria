(set-logic BV)

(synth-inv inv_fun ((k (_ BitVec 32))(j (_ BitVec 32))(i (_ BitVec 32))))

(define-fun pre_fun ((k (_ BitVec 32)) (j (_ BitVec 32)) (i (_ BitVec 32))) Bool
       ( and ( = i #x00000000 ) ( = j #x00000001 ) ( = k #x00000000 ) ))
(define-fun trans_fun ((k! (_ BitVec 32)) (j! (_ BitVec 32)) (i! (_ BitVec 32)) (k (_ BitVec 32)) (j (_ BitVec 32)) (i (_ BitVec 32))) Bool
       ( and ( bvult i #x000003e8 ) ( = i! ( bvadd i #x00000001 ) ) ( = j! ( bvadd j #x00000001 ) ) ( = k! ( bvadd k ( bvmul i! j! ) ) ) ))
(define-fun post_fun ((k (_ BitVec 32)) (j (_ BitVec 32)) (i (_ BitVec 32))) Bool
       ( or ( not ( bvule #x000003e8 i ) ) ( bvule ( bvmul #x000003e8 j ) k ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

