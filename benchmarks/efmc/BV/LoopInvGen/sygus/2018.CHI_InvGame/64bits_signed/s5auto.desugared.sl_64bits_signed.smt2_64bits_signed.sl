(set-logic BV)

(synth-inv inv_fun ((k (_ BitVec 64))(j (_ BitVec 64))(i (_ BitVec 64))))

(define-fun pre_fun ((k (_ BitVec 64)) (j (_ BitVec 64)) (i (_ BitVec 64))) Bool
       ( and ( = j #x0000000000000000 ) ( = i #x0000000000000000 ) ( bvsgt k #x0000000000000000 ) ))
(define-fun trans_fun ((k! (_ BitVec 64)) (j! (_ BitVec 64)) (i! (_ BitVec 64)) (k (_ BitVec 64)) (j (_ BitVec 64)) (i (_ BitVec 64))) Bool
       ( and ( bvslt j k ) ( = k! k ) ( = j! ( bvadd j #x0000000000000001 ) ) ( = i! ( bvadd i ( bvmul #x0000000000000002 k ) ) ) ))
(define-fun post_fun ((k (_ BitVec 64)) (j (_ BitVec 64)) (i (_ BitVec 64))) Bool
       ( or ( not ( bvsle k j ) ) ( = i ( bvmul #x0000000000000002 j k ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

