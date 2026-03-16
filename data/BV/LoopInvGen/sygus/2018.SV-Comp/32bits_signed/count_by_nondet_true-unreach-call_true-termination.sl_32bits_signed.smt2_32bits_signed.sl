(set-logic BV)

(synth-inv inv_fun ((j (_ BitVec 32))(k (_ BitVec 32))(i (_ BitVec 32))))

(define-fun pre_fun ((j (_ BitVec 32)) (k (_ BitVec 32)) (i (_ BitVec 32))) Bool
       ( and ( = i #x00000000 ) ( = k #x00000000 ) ))
(define-fun trans_fun ((j! (_ BitVec 32)) (k! (_ BitVec 32)) (i! (_ BitVec 32)) (j (_ BitVec 32)) (k (_ BitVec 32)) (i (_ BitVec 32))) Bool
       ( and ( bvslt i #x000f4240 ) ( bvsle #x00000001 j! ) ( bvslt j! #x000f4240 ) ( = i! ( bvadd i j! ) ) ( = k! ( bvadd k #x00000001 ) ) ))
(define-fun post_fun ((j (_ BitVec 32)) (k (_ BitVec 32)) (i (_ BitVec 32))) Bool
       ( or ( not ( bvsle #x000f4240 i ) ) ( bvsle k #x000f4240 ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

