(set-logic BV)

(synth-inv inv_fun ((z (_ BitVec 64))(y (_ BitVec 64))(x (_ BitVec 64))))

(define-fun pre_fun ((z (_ BitVec 64)) (y (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( and ( = x #x0000000000000000 ) ( = y #x0000000000000d05 ) ( = z #x0000000000001a0a ) ))
(define-fun trans_fun ((z! (_ BitVec 64)) (y! (_ BitVec 64)) (x! (_ BitVec 64)) (z (_ BitVec 64)) (y (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( and ( = x! ( bvadd x #x0000000000000001 ) ) ( = y! ( ite ( bvult x #x0000000000000d05 ) y ( bvadd y #x0000000000000001 ) ) ) ( = z! ( ite ( bvuge y #x0000000000001a0a ) ( bvadd z #x0000000000000001 ) z ) ) ))
(define-fun post_fun ((z (_ BitVec 64)) (y (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( not ( and ( = x #x000000000000270f ) ( = z x ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

