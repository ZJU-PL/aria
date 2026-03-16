(set-logic BV)

(synth-inv inv_fun ((z (_ BitVec 64))(x (_ BitVec 64))))

(define-fun pre_fun ((z (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( and ( bvugt x ( bvsub #x0000000000000000 #x0000000000000064 ) ) ( bvult x #x00000000000000c8 ) ( bvugt z #x0000000000000064 ) ( bvult z #x00000000000000c8 ) ))
(define-fun trans_fun ((z! (_ BitVec 64)) (x! (_ BitVec 64)) (z (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( let ( ( a!1 ( or ( and ( = x! ( bvadd x #x0000000000000001 ) ) ( = z! z ) ) ( and ( = x! ( bvsub x #x0000000000000001 ) ) ( = z! ( bvsub z #x0000000000000001 ) ) ) ) ) ) ( and ( bvult x #x0000000000000064 ) ( bvugt z #x0000000000000064 ) a!1 ) ))
(define-fun post_fun ((z (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( let ( ( a!1 ( not ( and ( = ( ( _ extract (_ bv63 64) (_ bv7 64) ) z ) #b000000000000000000000000000000000000000000000000000000000 ) ( bvule ( ( _ extract (_ bv6 64) (_ bv0 64) ) z ) #b1100100 ) ) ) ) ) ( or ( bvule #x0000000000000064 x ) ( and ( = ( ( _ extract (_ bv63 64) (_ bv7 64) ) z ) #b000000000000000000000000000000000000000000000000000000000 ) ( bvule ( ( _ extract (_ bv6 64) (_ bv0 64) ) z ) #b1100100 ) ) ( and ( not ( bvule #x0000000000000064 x ) ) a!1 ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

