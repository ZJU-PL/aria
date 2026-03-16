(set-logic BV)

(synth-inv inv_fun ((l (_ BitVec 32))(k (_ BitVec 32))(j (_ BitVec 32))(i (_ BitVec 32))))

(define-fun pre_fun ((l (_ BitVec 32)) (k (_ BitVec 32)) (j (_ BitVec 32)) (i (_ BitVec 32))) Bool
       ( and ( = j #x00000000 ) ( = i #x00000000 ) ( bvsge k #x00000000 ) ))
(define-fun trans_fun ((l! (_ BitVec 32)) (k! (_ BitVec 32)) (j! (_ BitVec 32)) (i! (_ BitVec 32)) (l (_ BitVec 32)) (k (_ BitVec 32)) (j (_ BitVec 32)) (i (_ BitVec 32))) Bool
       ( and ( bvslt j k ) ( = k! k ) ( = l! l ) ( = i! ( bvadd i ( bvmul l k ) ) ) ( = j! ( bvadd j #x00000001 ) ) ))
(define-fun post_fun ((l (_ BitVec 32)) (k (_ BitVec 32)) (j (_ BitVec 32)) (i (_ BitVec 32))) Bool
       ( or ( not ( bvsle k j ) ) ( = i ( bvmul j k l ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

