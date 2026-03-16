(set-logic BV)

(synth-inv inv_fun ((z (_ BitVec 64))(x (_ BitVec 64))(y (_ BitVec 64))))

(define-fun pre_fun ((z (_ BitVec 64)) (x (_ BitVec 64)) (y (_ BitVec 64))) Bool
       ( and ( = x #x0000000000000000 ) ( = y #x0000000000001f40 ) ( = z #x0000000000000000 ) ))
(define-fun trans_fun ((z! (_ BitVec 64)) (y! (_ BitVec 64)) (x! (_ BitVec 64)) (z (_ BitVec 64)) (x (_ BitVec 64)) (y (_ BitVec 64))) Bool
       ( and ( = x! ( bvadd x #x0000000000000001 ) ) ( = y! ( ite ( bvuge x #x0000000000001f40 ) ( bvadd y #x0000000000000001 ) ( bvsub y #x0000000000000001 ) ) ) ( = z! ( ite ( bvult x #x0000000000001f40 ) ( bvadd z #x0000000000000001 ) ( bvsub z #x0000000000000001 ) ) ) ))
(define-fun post_fun ((z (_ BitVec 64)) (x (_ BitVec 64)) (y (_ BitVec 64))) Bool
       ( not ( and ( = x #x0000000000003e80 ) ( = y #x0000000000001f40 ) ( = z #x0000000000000000 ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

