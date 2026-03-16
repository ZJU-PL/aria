(set-logic BV)

(synth-inv inv_fun ((w (_ BitVec 32))(x (_ BitVec 32))(y (_ BitVec 32))(z (_ BitVec 32))))

(define-fun pre_fun ((w (_ BitVec 32)) (x (_ BitVec 32)) (y (_ BitVec 32)) (z (_ BitVec 32))) Bool
       ( and ( = x #x00000000 ) ( = y #x00000000 ) ( = z #x00000000 ) ( = w #x00000000 ) ))
(define-fun trans_fun ((w! (_ BitVec 32)) (z! (_ BitVec 32)) (y! (_ BitVec 32)) (x! (_ BitVec 32)) (w (_ BitVec 32)) (x (_ BitVec 32)) (y (_ BitVec 32)) (z (_ BitVec 32))) Bool
       ( let ( ( a!1 ( ite ( bvugt ( bvsub y ( bvmul #x0000000a x ) ) #x00000000 ) ( bvadd z #x00000001 ) z ) ) ( a!2 ( ite ( bvugt ( bvsub y ( bvmul #x0000000a x ) ) #x00000000 ) w ( bvadd w #x00000001 ) ) ) ) ( and ( = x! ( bvadd x #x00000001 ) ) ( = y! ( bvadd y x ) ) ( = z! a!1 ) ( = w! a!2 ) ) ))
(define-fun post_fun ((w (_ BitVec 32)) (x (_ BitVec 32)) (y (_ BitVec 32)) (z (_ BitVec 32))) Bool
       ( let ( ( a!1 ( not ( and ( = ( ( _ extract (_ bv31 32) (_ bv7 32) ) x ) #b0000000000000000000000000 ) ( bvule ( ( _ extract (_ bv6 32) (_ bv0 32) ) x ) #b1100100 ) ) ) ) ) ( not ( and a!1 ( not ( bvule z w ) ) ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

