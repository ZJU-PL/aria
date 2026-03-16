(set-logic BV)

(synth-inv inv_fun ((k (_ BitVec 64))(j (_ BitVec 64))(i (_ BitVec 64))))

(define-fun pre_fun ((k (_ BitVec 64)) (j (_ BitVec 64)) (i (_ BitVec 64))) Bool
       ( and ( = i #x0000000000000000 ) ( bvugt j #x0000000000000000 ) ( bvugt k #x0000000000000000 ) ))
(define-fun trans_fun ((k! (_ BitVec 64)) (j! (_ BitVec 64)) (i! (_ BitVec 64)) (k (_ BitVec 64)) (j (_ BitVec 64)) (i (_ BitVec 64))) Bool
       ( and ( bvult i ( bvmul j k ) ) ( = i! ( bvadd i #x0000000000000001 ) ) ( = j! j ) ( = k! k ) ))
(define-fun post_fun ((k (_ BitVec 64)) (j (_ BitVec 64)) (i (_ BitVec 64))) Bool
       ( or ( not ( bvule ( bvmul j k ) i ) ) ( = i ( bvmul j k ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

