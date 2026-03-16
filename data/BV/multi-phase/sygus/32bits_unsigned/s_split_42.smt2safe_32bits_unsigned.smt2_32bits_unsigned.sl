(set-logic BV)

(synth-inv inv_fun ((z (_ BitVec 32))(y (_ BitVec 32))(x (_ BitVec 32))))

(define-fun pre_fun ((z (_ BitVec 32)) (y (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( and ( = x #x00000000 ) ( = y z ) ( = z #x00000000 ) ))
(define-fun trans_fun ((z! (_ BitVec 32)) (y! (_ BitVec 32)) (x! (_ BitVec 32)) (z (_ BitVec 32)) (y (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( and ( = x! ( bvadd x #x00000001 ) ) ( = y! ( ite ( bvuge x #x000006e5 ) ( bvadd y #x00000002 ) ( bvadd y #x00000001 ) ) ) ( = z! ( ite ( bvuge y #x00001685 ) ( bvadd z #x00000003 ) ( bvadd z #x00000002 ) ) ) ))
(define-fun post_fun ((z (_ BitVec 32)) (y (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( let ( ( a!1 ( not ( and ( = ( ( _ extract (_ bv31 32) (_ bv15 32) ) x ) #b00000000000000000 ) ( bvule ( ( _ extract (_ bv14 32) (_ bv0 32) ) x ) #b100010011110010 ) ) ) ) ) ( not ( and a!1 ( = ( ( _ extract (_ bv31 32) (_ bv15 32) ) z ) #b00000000000000000 ) ( bvule ( ( _ extract (_ bv14 32) (_ bv0 32) ) z ) #b110110000000010 ) ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

