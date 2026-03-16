(set-logic BV)

(synth-inv inv_fun ((z (_ BitVec 32))(y (_ BitVec 32))(x (_ BitVec 32))))

(define-fun pre_fun ((z (_ BitVec 32)) (y (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( and ( = x #x00000000 ) ( = y #x00000d05 ) ( = z #x00001a0a ) ))
(define-fun trans_fun ((z! (_ BitVec 32)) (y! (_ BitVec 32)) (x! (_ BitVec 32)) (z (_ BitVec 32)) (y (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( and ( = x! ( bvadd x #x00000001 ) ) ( = y! ( ite ( bvult x #x00000d05 ) y ( bvadd y #x00000001 ) ) ) ( = z! ( ite ( bvuge y #x00001a0a ) ( bvadd z #x00000001 ) z ) ) ))
(define-fun post_fun ((z (_ BitVec 32)) (y (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( not ( and ( = x #x0000270f ) ( = z x ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

