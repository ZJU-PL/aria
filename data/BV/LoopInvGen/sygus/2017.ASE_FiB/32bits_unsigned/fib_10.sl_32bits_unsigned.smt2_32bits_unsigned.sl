(set-logic BV)

(synth-inv inv_fun ((z (_ BitVec 32))(w (_ BitVec 32))(y (_ BitVec 32))(x (_ BitVec 32))))

(define-fun pre_fun ((z (_ BitVec 32)) (w (_ BitVec 32)) (y (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( and ( = w #x00000001 ) ( = z #x00000000 ) ( = x #x00000000 ) ( = y #x00000000 ) ))
(define-fun trans_fun ((z! (_ BitVec 32)) (w! (_ BitVec 32)) (y! (_ BitVec 32)) (x! (_ BitVec 32)) (z (_ BitVec 32)) (w (_ BitVec 32)) (y (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( let ( ( a!1 ( or ( and ( = w #x00000001 ) ( = z #x00000000 ) ( = x! ( bvadd x #x00000001 ) ) ( = w! #x00000000 ) ( = y! ( bvadd y #x00000001 ) ) ( = z! #x00000001 ) ) ( and ( not ( = w #x00000001 ) ) ( = z #x00000000 ) ( = x! x ) ( = w! w ) ( = y! ( bvadd y #x00000001 ) ) ( = z! #x00000001 ) ) ( and ( = w #x00000001 ) ( not ( = z #x00000000 ) ) ( = x! ( bvadd x #x00000001 ) ) ( = w! #x00000000 ) ( = y! y ) ( = z! z ) ) ( and ( not ( = w #x00000001 ) ) ( not ( = z #x00000000 ) ) ( = x! x ) ( = w! w ) ( = y! y ) ( = z! z ) ) ) ) ) ( and a!1 ) ))
(define-fun post_fun ((z (_ BitVec 32)) (w (_ BitVec 32)) (y (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( = x y ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

