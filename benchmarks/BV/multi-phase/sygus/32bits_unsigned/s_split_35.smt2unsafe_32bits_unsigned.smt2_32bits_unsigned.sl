(set-logic BV)

(synth-inv inv_fun ((w (_ BitVec 32))(z (_ BitVec 32))(x (_ BitVec 32))(y (_ BitVec 32))))

(define-fun pre_fun ((w (_ BitVec 32)) (z (_ BitVec 32)) (x (_ BitVec 32)) (y (_ BitVec 32))) Bool
       ( and ( = x #x00000000 ) ( bvugt y z ) ( = w #x00000000 ) ))
(define-fun trans_fun ((w! (_ BitVec 32)) (z! (_ BitVec 32)) (y! (_ BitVec 32)) (x! (_ BitVec 32)) (w (_ BitVec 32)) (z (_ BitVec 32)) (x (_ BitVec 32)) (y (_ BitVec 32))) Bool
       ( and ( = x! ( bvadd #x00000005 x ) ) ( = y! ( bvadd #x00000003 y ) ) ( = z! ( bvadd #x00000001 z ) ) ( = w! ( ite ( bvult x z ) ( bvadd w #x00000001 ) ( bvsub w #x00000001 ) ) ) ))
(define-fun post_fun ((w (_ BitVec 32)) (z (_ BitVec 32)) (x (_ BitVec 32)) (y (_ BitVec 32))) Bool
       ( not ( and ( not ( bvule x y ) ) ( = w #x00000000 ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

