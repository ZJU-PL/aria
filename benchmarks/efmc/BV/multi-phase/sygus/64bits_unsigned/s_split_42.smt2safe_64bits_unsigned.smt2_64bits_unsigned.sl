(set-logic BV)

(synth-inv inv_fun ((z (_ BitVec 64))(y (_ BitVec 64))(x (_ BitVec 64))))

(define-fun pre_fun ((z (_ BitVec 64)) (y (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( and ( = x #x0000000000000000 ) ( = y z ) ( = z #x0000000000000000 ) ))
(define-fun trans_fun ((z! (_ BitVec 64)) (y! (_ BitVec 64)) (x! (_ BitVec 64)) (z (_ BitVec 64)) (y (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( and ( = x! ( bvadd x #x0000000000000001 ) ) ( = y! ( ite ( bvuge x #x00000000000006e5 ) ( bvadd y #x0000000000000002 ) ( bvadd y #x0000000000000001 ) ) ) ( = z! ( ite ( bvuge y #x0000000000001685 ) ( bvadd z #x0000000000000003 ) ( bvadd z #x0000000000000002 ) ) ) ))
(define-fun post_fun ((z (_ BitVec 64)) (y (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( let ( ( a!1 ( not ( and ( = ( ( _ extract (_ bv63 64) (_ bv15 64) ) x ) #b0000000000000000000000000000000000000000000000000 ) ( bvule ( ( _ extract (_ bv14 64) (_ bv0 64) ) x ) #b100010011110010 ) ) ) ) ) ( not ( and a!1 ( = ( ( _ extract (_ bv63 64) (_ bv15 64) ) z ) #b0000000000000000000000000000000000000000000000000 ) ( bvule ( ( _ extract (_ bv14 64) (_ bv0 64) ) z ) #b110110000000010 ) ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

