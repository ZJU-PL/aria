(set-logic BV)

(synth-inv inv_fun ((w (_ BitVec 64))(z (_ BitVec 64))(x (_ BitVec 64))(v (_ BitVec 64))))

(define-fun pre_fun ((w (_ BitVec 64)) (z (_ BitVec 64)) (x (_ BitVec 64)) (v (_ BitVec 64))) Bool
       ( and ( bvugt x z ) ( = v #x0000000000000000 ) ( = w #x0000000000000000 ) ))
(define-fun trans_fun ((w! (_ BitVec 64)) (v! (_ BitVec 64)) (z! (_ BitVec 64)) (x! (_ BitVec 64)) (w (_ BitVec 64)) (z (_ BitVec 64)) (x (_ BitVec 64)) (v (_ BitVec 64))) Bool
       ( and ( = x! ( bvadd x #x0000000000000001 ) ) ( = z! ( bvadd z #x0000000000000002 ) ) ( = v! ( ite ( bvult x z ) ( bvadd v #x0000000000000001 ) v ) ) ( = w! ( ite ( bvult x z ) w ( bvadd w #x0000000000000001 ) ) ) ))
(define-fun post_fun ((w (_ BitVec 64)) (z (_ BitVec 64)) (x (_ BitVec 64)) (v (_ BitVec 64))) Bool
       ( let ( ( a!1 ( not ( and ( = ( ( _ extract (_ bv63 64) (_ bv10 64) ) v ) #b000000000000000000000000000000000000000000000000000000 ) ( bvule ( ( _ extract (_ bv9 64) (_ bv0 64) ) v ) #b1111101000 ) ) ) ) ) ( not ( and a!1 ( = w #x0000000000000000 ) ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

