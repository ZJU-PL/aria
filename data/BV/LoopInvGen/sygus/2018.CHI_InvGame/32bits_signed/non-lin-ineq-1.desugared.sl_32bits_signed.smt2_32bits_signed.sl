(set-logic BV)

(synth-inv inv_fun ((k (_ BitVec 32))(j (_ BitVec 32))(i (_ BitVec 32))))

(define-fun pre_fun ((k (_ BitVec 32)) (j (_ BitVec 32)) (i (_ BitVec 32))) Bool
       ( and ( = i #x00000000 ) ( bvsle ( bvmul i j ) k ) ))
(define-fun trans_fun ((k! (_ BitVec 32)) (j! (_ BitVec 32)) (i! (_ BitVec 32)) (k (_ BitVec 32)) (j (_ BitVec 32)) (i (_ BitVec 32))) Bool
       ( and ( bvslt j #x000003e8 ) ( = i! ( bvadd i #x00000001 ) ) ( = j! j ) ( = k! ( bvadd k j ) ) ))
(define-fun post_fun ((k (_ BitVec 32)) (j (_ BitVec 32)) (i (_ BitVec 32))) Bool
       ( or ( not ( bvsle #x000003e8 j ) ) ( bvsle ( bvmul i j ) k ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

