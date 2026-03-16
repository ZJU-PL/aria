(set-logic BV)

(synth-inv inv_fun ((k (_ BitVec 32))(j (_ BitVec 32))(i (_ BitVec 32))))

(define-fun pre_fun ((k (_ BitVec 32)) (j (_ BitVec 32)) (i (_ BitVec 32))) Bool
       ( and ( = j #x00000000 ) ( = i #x00000000 ) ( bvsgt k #x00000000 ) ))
(define-fun trans_fun ((k! (_ BitVec 32)) (j! (_ BitVec 32)) (i! (_ BitVec 32)) (k (_ BitVec 32)) (j (_ BitVec 32)) (i (_ BitVec 32))) Bool
       ( and ( bvslt j k ) ( = k! k ) ( = j! ( bvadd j #x00000001 ) ) ( = i! ( bvadd i ( bvmul #x00000002 k ) ) ) ))
(define-fun post_fun ((k (_ BitVec 32)) (j (_ BitVec 32)) (i (_ BitVec 32))) Bool
       ( or ( not ( bvsle k j ) ) ( = i ( bvmul #x00000002 j k ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

