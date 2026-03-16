(set-logic BV)

(synth-inv inv_fun ((w (_ BitVec 32))(z (_ BitVec 32))(x (_ BitVec 32))(v (_ BitVec 32))))

(define-fun pre_fun ((w (_ BitVec 32)) (z (_ BitVec 32)) (x (_ BitVec 32)) (v (_ BitVec 32))) Bool
       ( and ( bvugt x z ) ( = v #x00000000 ) ( = w #x00000000 ) ))
(define-fun trans_fun ((w! (_ BitVec 32)) (v! (_ BitVec 32)) (z! (_ BitVec 32)) (x! (_ BitVec 32)) (w (_ BitVec 32)) (z (_ BitVec 32)) (x (_ BitVec 32)) (v (_ BitVec 32))) Bool
       ( and ( = x! ( bvadd x #x00000001 ) ) ( = z! ( bvadd z #x00000002 ) ) ( = v! ( ite ( bvult x z ) ( bvadd v #x00000001 ) v ) ) ( = w! ( ite ( bvult x z ) w ( bvadd w #x00000001 ) ) ) ))
(define-fun post_fun ((w (_ BitVec 32)) (z (_ BitVec 32)) (x (_ BitVec 32)) (v (_ BitVec 32))) Bool
       ( let ( ( a!1 ( not ( and ( = ( ( _ extract (_ bv31 32) (_ bv10 32) ) v ) #b0000000000000000000000 ) ( bvule ( ( _ extract (_ bv9 32) (_ bv0 32) ) v ) #b1111101000 ) ) ) ) ) ( not ( and a!1 ( not ( = w #x00000000 ) ) ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

