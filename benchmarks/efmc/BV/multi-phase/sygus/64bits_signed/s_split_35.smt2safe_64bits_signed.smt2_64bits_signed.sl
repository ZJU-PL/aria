(set-logic BV)

(synth-inv inv_fun ((w (_ BitVec 64))(z (_ BitVec 64))(x (_ BitVec 64))(y (_ BitVec 64))))

(define-fun pre_fun ((w (_ BitVec 64)) (z (_ BitVec 64)) (x (_ BitVec 64)) (y (_ BitVec 64))) Bool
       ( and ( = x #x0000000000000000 ) ( bvsgt y z ) ( = w #x0000000000000000 ) ))
(define-fun trans_fun ((w! (_ BitVec 64)) (z! (_ BitVec 64)) (y! (_ BitVec 64)) (x! (_ BitVec 64)) (w (_ BitVec 64)) (z (_ BitVec 64)) (x (_ BitVec 64)) (y (_ BitVec 64))) Bool
       ( and ( = x! ( bvadd #x0000000000000005 x ) ) ( = y! ( bvadd #x0000000000000003 y ) ) ( = z! ( bvadd #x0000000000000001 z ) ) ( = w! ( ite ( bvslt x z ) ( bvadd w #x0000000000000001 ) ( bvsub w #x0000000000000001 ) ) ) ))
(define-fun post_fun ((w (_ BitVec 64)) (z (_ BitVec 64)) (x (_ BitVec 64)) (y (_ BitVec 64))) Bool
       ( not ( and ( not ( bvsle x y ) ) ( not ( bvsle w #x0000000000000000 ) ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

