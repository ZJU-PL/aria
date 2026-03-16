(set-logic BV)

(synth-inv inv_fun ((w (_ BitVec 32))(z (_ BitVec 32))(x (_ BitVec 32))(v (_ BitVec 32))))

(define-fun pre_fun ((w (_ BitVec 32)) (z (_ BitVec 32)) (x (_ BitVec 32)) (v (_ BitVec 32))) Bool
       ( and ( bvsgt x z ) ( = v #x00000000 ) ( = w #x00000000 ) ))
(define-fun trans_fun ((w! (_ BitVec 32)) (v! (_ BitVec 32)) (z! (_ BitVec 32)) (x! (_ BitVec 32)) (w (_ BitVec 32)) (z (_ BitVec 32)) (x (_ BitVec 32)) (v (_ BitVec 32))) Bool
       ( and ( = x! ( bvadd x #x00000001 ) ) ( = z! ( bvadd z #x00000002 ) ) ( = v! ( ite ( bvslt x z ) ( bvadd v #x00000001 ) v ) ) ( = w! ( ite ( bvslt x z ) w ( bvadd w #x00000001 ) ) ) ))
(define-fun post_fun ((w (_ BitVec 32)) (z (_ BitVec 32)) (x (_ BitVec 32)) (v (_ BitVec 32))) Bool
       ( not ( and ( not ( bvsle v #x000003e8 ) ) ( bvsle w #x00000000 ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

