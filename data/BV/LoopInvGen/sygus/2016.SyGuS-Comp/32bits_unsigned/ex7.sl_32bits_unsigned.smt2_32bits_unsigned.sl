(set-logic BV)

(synth-inv inv_fun ((i (_ BitVec 32))(y (_ BitVec 32))(x (_ BitVec 32))))

(define-fun pre_fun ((i (_ BitVec 32)) (y (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( and ( = i #x00000000 ) ( bvuge x #x00000000 ) ( bvuge y #x00000000 ) ( bvuge x y ) ))
(define-fun trans_fun ((i! (_ BitVec 32)) (y! (_ BitVec 32)) (x! (_ BitVec 32)) (i (_ BitVec 32)) (y (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( and ( bvult i y ) ( = i! ( bvadd i #x00000001 ) ) ( = y! y ) ( = x! x ) ))
(define-fun post_fun ((i (_ BitVec 32)) (y (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( not ( and ( not ( bvule y i ) ) ( bvule x i ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

