(set-logic BV)

(synth-inv inv_fun ((j (_ BitVec 64))(k (_ BitVec 64))(i (_ BitVec 64))))

(define-fun pre_fun ((j (_ BitVec 64)) (k (_ BitVec 64)) (i (_ BitVec 64))) Bool
       ( and ( = i #x0000000000000000 ) ( = k #x0000000000000000 ) ))
(define-fun trans_fun ((j! (_ BitVec 64)) (k! (_ BitVec 64)) (i! (_ BitVec 64)) (j (_ BitVec 64)) (k (_ BitVec 64)) (i (_ BitVec 64))) Bool
       ( and ( bvslt i #x00000000000f4240 ) ( bvsle #x0000000000000001 j! ) ( bvslt j! #x00000000000f4240 ) ( = i! ( bvadd i j! ) ) ( = k! ( bvadd k #x0000000000000001 ) ) ))
(define-fun post_fun ((j (_ BitVec 64)) (k (_ BitVec 64)) (i (_ BitVec 64))) Bool
       ( or ( not ( bvsle #x00000000000f4240 i ) ) ( bvsle k #x00000000000f4240 ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

