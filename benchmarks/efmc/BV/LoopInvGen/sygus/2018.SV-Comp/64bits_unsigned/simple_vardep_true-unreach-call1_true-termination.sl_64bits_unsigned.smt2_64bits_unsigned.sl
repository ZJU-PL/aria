(set-logic BV)

(synth-inv inv_fun ((k (_ BitVec 64))(j (_ BitVec 64))(i (_ BitVec 64))))

(define-fun pre_fun ((k (_ BitVec 64)) (j (_ BitVec 64)) (i (_ BitVec 64))) Bool
       ( and ( = i #x0000000000000000 ) ( = j #x0000000000000000 ) ( = k #x0000000000000000 ) ))
(define-fun trans_fun ((k! (_ BitVec 64)) (j! (_ BitVec 64)) (i! (_ BitVec 64)) (k (_ BitVec 64)) (j (_ BitVec 64)) (i (_ BitVec 64))) Bool
       ( and ( bvult k #x000000000fffffff ) ( = i! ( bvadd i #x0000000000000001 ) ) ( = j! ( bvadd j #x0000000000000002 ) ) ( = k! ( bvadd k #x0000000000000003 ) ) ))
(define-fun post_fun ((k (_ BitVec 64)) (j (_ BitVec 64)) (i (_ BitVec 64))) Bool
       ( or ( not ( bvule #x000000000fffffff k ) ) ( = k ( bvadd i j ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

