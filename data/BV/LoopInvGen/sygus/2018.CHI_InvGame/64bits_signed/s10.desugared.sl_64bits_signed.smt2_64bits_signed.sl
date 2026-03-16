(set-logic BV)

(synth-inv inv_fun ((k (_ BitVec 64))(j (_ BitVec 64))(i (_ BitVec 64))))

(define-fun pre_fun ((k (_ BitVec 64)) (j (_ BitVec 64)) (i (_ BitVec 64))) Bool
       ( and ( = j #x0000000000000000 ) ( = i #x000000000000000a ) ))
(define-fun trans_fun ((k! (_ BitVec 64)) (j! (_ BitVec 64)) (i! (_ BitVec 64)) (k (_ BitVec 64)) (j (_ BitVec 64)) (i (_ BitVec 64))) Bool
       ( and ( bvslt j #x00000000000003e8 ) ( = i! ( bvadd i k ) ) ( = j! ( bvadd j #x0000000000000001 ) ) ( = k! k ) ))
(define-fun post_fun ((k (_ BitVec 64)) (j (_ BitVec 64)) (i (_ BitVec 64))) Bool
       ( or ( not ( bvsle #x00000000000003e8 j ) ) ( = i ( bvadd #x000000000000000a ( bvmul j k ) ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

