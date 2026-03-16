(set-logic BV)

(synth-inv inv_fun ((i (_ BitVec 64))(c (_ BitVec 64))))

(define-fun pre_fun ((i (_ BitVec 64)) (c (_ BitVec 64))) Bool
       ( = i #x0000000000000000 ))
(define-fun trans_fun ((i! (_ BitVec 64)) (c! (_ BitVec 64)) (i (_ BitVec 64)) (c (_ BitVec 64))) Bool
       ( and ( bvsgt c #x0000000000000030 ) ( bvslt c #x0000000000000039 ) ( = i! ( bvadd i i ( bvsub c #x0000000000000030 ) ) ) ))
(define-fun post_fun ((i (_ BitVec 64)) (c (_ BitVec 64))) Bool
       ( or ( bvsle #x0000000000000000 i ) ( not ( bvsle c #x0000000000000039 ) ) ( not ( bvsle #x0000000000000030 c ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

