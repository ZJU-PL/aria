(set-logic BV)

(synth-inv inv_fun ((l (_ BitVec 64))(k (_ BitVec 64))(j (_ BitVec 64))(i (_ BitVec 64))))

(define-fun pre_fun ((l (_ BitVec 64)) (k (_ BitVec 64)) (j (_ BitVec 64)) (i (_ BitVec 64))) Bool
       ( and ( = j #x0000000000000000 ) ( = i #x0000000000000000 ) ( bvsge k #x0000000000000000 ) ))
(define-fun trans_fun ((l! (_ BitVec 64)) (k! (_ BitVec 64)) (j! (_ BitVec 64)) (i! (_ BitVec 64)) (l (_ BitVec 64)) (k (_ BitVec 64)) (j (_ BitVec 64)) (i (_ BitVec 64))) Bool
       ( and ( bvslt j k ) ( = k! k ) ( = l! l ) ( = i! ( bvadd i ( bvmul l k ) ) ) ( = j! ( bvadd j #x0000000000000001 ) ) ))
(define-fun post_fun ((l (_ BitVec 64)) (k (_ BitVec 64)) (j (_ BitVec 64)) (i (_ BitVec 64))) Bool
       ( or ( not ( bvsle k j ) ) ( = i ( bvmul j k l ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

