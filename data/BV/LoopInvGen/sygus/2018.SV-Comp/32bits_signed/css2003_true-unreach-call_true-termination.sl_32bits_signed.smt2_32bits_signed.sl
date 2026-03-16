(set-logic BV)

(synth-inv inv_fun ((k (_ BitVec 32))(j (_ BitVec 32))(i (_ BitVec 32))))

(define-fun pre_fun ((k (_ BitVec 32)) (j (_ BitVec 32)) (i (_ BitVec 32))) Bool
       ( and ( = i #x00000001 ) ( = j #x00000001 ) ( bvsle #x00000000 k ) ( bvsle k #x00000001 ) ))
(define-fun trans_fun ((k! (_ BitVec 32)) (j! (_ BitVec 32)) (i! (_ BitVec 32)) (k (_ BitVec 32)) (j (_ BitVec 32)) (i (_ BitVec 32))) Bool
       ( and ( bvslt i #x000f4240 ) ( = i! ( bvadd i #x00000001 ) ) ( = j! ( bvadd j k ) ) ( = k! ( bvsub k #x00000001 ) ) ))
(define-fun post_fun ((k (_ BitVec 32)) (j (_ BitVec 32)) (i (_ BitVec 32))) Bool
       ( or ( bvsle #x000f4240 i ) ( and ( bvsle #x00000001 ( bvadd i k ) ) ( bvsle ( bvadd i k ) #x00000002 ) ( bvsle #x00000001 i ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

