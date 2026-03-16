(set-logic BV)

(synth-inv inv_fun ((k (_ BitVec 64))(j (_ BitVec 64))(i (_ BitVec 64))))

(define-fun pre_fun ((k (_ BitVec 64)) (j (_ BitVec 64)) (i (_ BitVec 64))) Bool
       ( and ( = i #x0000000000000000 ) ( = j #x0000000000000001 ) ( = k #x0000000000000000 ) ))
(define-fun trans_fun ((k! (_ BitVec 64)) (j! (_ BitVec 64)) (i! (_ BitVec 64)) (k (_ BitVec 64)) (j (_ BitVec 64)) (i (_ BitVec 64))) Bool
       ( and ( bvslt i #x00000000000003e8 ) ( = i! ( bvadd i #x0000000000000001 ) ) ( = j! ( bvadd j #x0000000000000001 ) ) ( = k! ( bvadd k ( bvmul i! j! ) ) ) ))
(define-fun post_fun ((k (_ BitVec 64)) (j (_ BitVec 64)) (i (_ BitVec 64))) Bool
       ( or ( not ( bvsle #x00000000000003e8 i ) ) ( bvsle ( bvmul #x00000000000003e8 j ) k ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

