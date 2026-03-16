(set-logic BV)

(synth-inv inv_fun ((k (_ BitVec 64))(j (_ BitVec 64))(i (_ BitVec 64))))

(define-fun pre_fun ((k (_ BitVec 64)) (j (_ BitVec 64)) (i (_ BitVec 64))) Bool
       ( and ( = i #x0000000000000001 ) ( = j #x0000000000000001 ) ( bvsle #x0000000000000000 k ) ( bvsle k #x0000000000000001 ) ))
(define-fun trans_fun ((k! (_ BitVec 64)) (j! (_ BitVec 64)) (i! (_ BitVec 64)) (k (_ BitVec 64)) (j (_ BitVec 64)) (i (_ BitVec 64))) Bool
       ( and ( bvslt i #x00000000000f4240 ) ( = i! ( bvadd i #x0000000000000001 ) ) ( = j! ( bvadd j k ) ) ( = k! ( bvsub k #x0000000000000001 ) ) ))
(define-fun post_fun ((k (_ BitVec 64)) (j (_ BitVec 64)) (i (_ BitVec 64))) Bool
       ( or ( bvsle #x00000000000f4240 i ) ( and ( bvsle #x0000000000000001 ( bvadd i k ) ) ( bvsle ( bvadd i k ) #x0000000000000002 ) ( bvsle #x0000000000000001 i ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

