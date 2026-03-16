(set-logic BV)

(synth-inv inv_fun ((i (_ BitVec 64))(y (_ BitVec 64))(x (_ BitVec 64))))

(define-fun pre_fun ((i (_ BitVec 64)) (y (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( and ( = i #x0000000000000000 ) ( bvuge x #x0000000000000000 ) ( bvuge y #x0000000000000000 ) ( bvuge x y ) ))
(define-fun trans_fun ((i! (_ BitVec 64)) (y! (_ BitVec 64)) (x! (_ BitVec 64)) (i (_ BitVec 64)) (y (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( and ( bvult i y ) ( = i! ( bvadd i #x0000000000000001 ) ) ( = y! y ) ( = x! x ) ))
(define-fun post_fun ((i (_ BitVec 64)) (y (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( not ( and ( not ( bvule y i ) ) ( bvule x i ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

