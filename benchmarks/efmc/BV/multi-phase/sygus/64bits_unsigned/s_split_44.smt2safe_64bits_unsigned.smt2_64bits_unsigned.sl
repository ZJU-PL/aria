(set-logic BV)

(synth-inv inv_fun ((z (_ BitVec 64))(y (_ BitVec 64))(x (_ BitVec 64))))

(define-fun pre_fun ((z (_ BitVec 64)) (y (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( and ( = x #x0000000000000000 ) ( = y #x00000000000003e8 ) ( = z #x00000000000007d0 ) ))
(define-fun trans_fun ((z! (_ BitVec 64)) (y! (_ BitVec 64)) (x! (_ BitVec 64)) (z (_ BitVec 64)) (y (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( and ( = x! ( bvadd #x0000000000000001 x ) ) ( = y! ( ite ( bvuge x #x00000000000003e8 ) ( bvadd y #x0000000000000001 ) y ) ) ( = z! ( ite ( bvuge y #x00000000000007d0 ) ( bvadd z #x0000000000000001 ) z ) ) ))
(define-fun post_fun ((z (_ BitVec 64)) (y (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( not ( and ( bvule #x00000000000007d0 y ) ( not ( = x z ) ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

