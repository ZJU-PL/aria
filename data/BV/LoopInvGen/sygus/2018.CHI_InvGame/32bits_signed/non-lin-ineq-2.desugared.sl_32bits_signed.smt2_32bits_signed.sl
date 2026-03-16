(set-logic BV)

(synth-inv inv_fun ((k (_ BitVec 32))(j (_ BitVec 32))(i (_ BitVec 32))))

(define-fun pre_fun ((k (_ BitVec 32)) (j (_ BitVec 32)) (i (_ BitVec 32))) Bool
       ( and ( = i #x00000000 ) ( bvsgt j #x00000000 ) ( bvsgt k #x00000000 ) ))
(define-fun trans_fun ((k! (_ BitVec 32)) (j! (_ BitVec 32)) (i! (_ BitVec 32)) (k (_ BitVec 32)) (j (_ BitVec 32)) (i (_ BitVec 32))) Bool
       ( and ( bvslt i ( bvmul j k ) ) ( = i! ( bvadd i #x00000001 ) ) ( = j! j ) ( = k! k ) ))
(define-fun post_fun ((k (_ BitVec 32)) (j (_ BitVec 32)) (i (_ BitVec 32))) Bool
       ( or ( not ( bvsle ( bvmul j k ) i ) ) ( = i ( bvmul j k ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

