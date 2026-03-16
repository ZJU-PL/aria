(set-logic BV)

(synth-inv inv_fun ((l (_ BitVec 32))(k (_ BitVec 32))(j (_ BitVec 32))(i (_ BitVec 32))))

(define-fun pre_fun ((l (_ BitVec 32)) (k (_ BitVec 32)) (j (_ BitVec 32)) (i (_ BitVec 32))) Bool
       ( and ( = j #x00000000 ) ( = i l ) ))
(define-fun trans_fun ((l! (_ BitVec 32)) (k! (_ BitVec 32)) (j! (_ BitVec 32)) (i! (_ BitVec 32)) (l (_ BitVec 32)) (k (_ BitVec 32)) (j (_ BitVec 32)) (i (_ BitVec 32))) Bool
       ( and ( bvslt j #x000003e8 ) ( = i! ( bvadd i k ) ) ( = j! ( bvadd j #x00000001 ) ) ( = k! k ) ( = l! l ) ))
(define-fun post_fun ((l (_ BitVec 32)) (k (_ BitVec 32)) (j (_ BitVec 32)) (i (_ BitVec 32))) Bool
       ( or ( not ( bvsle #x000003e8 j ) ) ( = i ( bvadd l ( bvmul j k ) ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

