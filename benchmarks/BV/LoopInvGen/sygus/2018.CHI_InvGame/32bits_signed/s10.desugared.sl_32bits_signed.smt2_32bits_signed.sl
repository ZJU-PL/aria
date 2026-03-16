(set-logic BV)

(synth-inv inv_fun ((k (_ BitVec 32))(j (_ BitVec 32))(i (_ BitVec 32))))

(define-fun pre_fun ((k (_ BitVec 32)) (j (_ BitVec 32)) (i (_ BitVec 32))) Bool
       ( and ( = j #x00000000 ) ( = i #x0000000a ) ))
(define-fun trans_fun ((k! (_ BitVec 32)) (j! (_ BitVec 32)) (i! (_ BitVec 32)) (k (_ BitVec 32)) (j (_ BitVec 32)) (i (_ BitVec 32))) Bool
       ( and ( bvslt j #x000003e8 ) ( = i! ( bvadd i k ) ) ( = j! ( bvadd j #x00000001 ) ) ( = k! k ) ))
(define-fun post_fun ((k (_ BitVec 32)) (j (_ BitVec 32)) (i (_ BitVec 32))) Bool
       ( or ( not ( bvsle #x000003e8 j ) ) ( = i ( bvadd #x0000000a ( bvmul j k ) ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

