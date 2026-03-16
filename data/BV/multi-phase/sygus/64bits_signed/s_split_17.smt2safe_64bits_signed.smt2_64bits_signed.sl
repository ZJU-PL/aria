(set-logic BV)

(synth-inv inv_fun ((w (_ BitVec 64))(z (_ BitVec 64))(x (_ BitVec 64))(v (_ BitVec 64))))

(define-fun pre_fun ((w (_ BitVec 64)) (z (_ BitVec 64)) (x (_ BitVec 64)) (v (_ BitVec 64))) Bool
       ( and ( bvsgt x z ) ( = v #x0000000000000000 ) ( = w #x0000000000000000 ) ))
(define-fun trans_fun ((w! (_ BitVec 64)) (v! (_ BitVec 64)) (z! (_ BitVec 64)) (x! (_ BitVec 64)) (w (_ BitVec 64)) (z (_ BitVec 64)) (x (_ BitVec 64)) (v (_ BitVec 64))) Bool
       ( and ( = x! ( bvadd x #x0000000000000001 ) ) ( = z! ( bvadd z #x0000000000000002 ) ) ( = v! ( ite ( bvslt x z ) ( bvadd v #x0000000000000001 ) v ) ) ( = w! ( ite ( bvslt x z ) w ( bvadd w #x0000000000000001 ) ) ) ))
(define-fun post_fun ((w (_ BitVec 64)) (z (_ BitVec 64)) (x (_ BitVec 64)) (v (_ BitVec 64))) Bool
       ( not ( and ( not ( bvsle v #x00000000000003e8 ) ) ( bvsle w #x0000000000000000 ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

