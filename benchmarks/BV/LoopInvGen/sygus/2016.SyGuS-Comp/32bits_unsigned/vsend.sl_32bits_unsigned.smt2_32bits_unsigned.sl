(set-logic BV)

(synth-inv inv_fun ((i (_ BitVec 32))(c (_ BitVec 32))))

(define-fun pre_fun ((i (_ BitVec 32)) (c (_ BitVec 32))) Bool
       ( = i #x00000000 ))
(define-fun trans_fun ((i! (_ BitVec 32)) (c! (_ BitVec 32)) (i (_ BitVec 32)) (c (_ BitVec 32))) Bool
       ( and ( bvugt c #x00000030 ) ( bvult c #x00000039 ) ( = i! ( bvadd i i ( bvsub c #x00000030 ) ) ) ))
(define-fun post_fun ((i (_ BitVec 32)) (c (_ BitVec 32))) Bool
       ( t r u e ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

