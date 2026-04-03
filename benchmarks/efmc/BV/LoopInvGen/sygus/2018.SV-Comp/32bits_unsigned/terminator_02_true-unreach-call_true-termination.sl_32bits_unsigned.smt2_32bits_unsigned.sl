(set-logic BV)

(synth-inv inv_fun ((z (_ BitVec 32))(x (_ BitVec 32))))

(define-fun pre_fun ((z (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( and ( bvugt x ( bvsub #x00000000 #x00000064 ) ) ( bvult x #x000000c8 ) ( bvugt z #x00000064 ) ( bvult z #x000000c8 ) ))
(define-fun trans_fun ((z! (_ BitVec 32)) (x! (_ BitVec 32)) (z (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( let ( ( a!1 ( or ( and ( = x! ( bvadd x #x00000001 ) ) ( = z! z ) ) ( and ( = x! ( bvsub x #x00000001 ) ) ( = z! ( bvsub z #x00000001 ) ) ) ) ) ) ( and ( bvult x #x00000064 ) ( bvugt z #x00000064 ) a!1 ) ))
(define-fun post_fun ((z (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( let ( ( a!1 ( not ( and ( = ( ( _ extract (_ bv31 32) (_ bv7 32) ) z ) #b0000000000000000000000000 ) ( bvule ( ( _ extract (_ bv6 32) (_ bv0 32) ) z ) #b1100100 ) ) ) ) ) ( or ( bvule #x00000064 x ) ( and ( = ( ( _ extract (_ bv31 32) (_ bv7 32) ) z ) #b0000000000000000000000000 ) ( bvule ( ( _ extract (_ bv6 32) (_ bv0 32) ) z ) #b1100100 ) ) ( and ( not ( bvule #x00000064 x ) ) a!1 ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

