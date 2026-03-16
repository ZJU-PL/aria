(set-logic BV)

(synth-inv inv_fun ((z (_ BitVec 64))(w (_ BitVec 64))(y (_ BitVec 64))(x (_ BitVec 64))))

(define-fun pre_fun ((z (_ BitVec 64)) (w (_ BitVec 64)) (y (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( and ( = w #x0000000000000001 ) ( = z #x0000000000000000 ) ( = x #x0000000000000000 ) ( = y #x0000000000000000 ) ))
(define-fun trans_fun ((z! (_ BitVec 64)) (w! (_ BitVec 64)) (y! (_ BitVec 64)) (x! (_ BitVec 64)) (z (_ BitVec 64)) (w (_ BitVec 64)) (y (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( let ( ( a!1 ( or ( and ( = w #x0000000000000001 ) ( = z #x0000000000000000 ) ( = x! ( bvadd x #x0000000000000001 ) ) ( = w! #x0000000000000000 ) ( = y! ( bvadd y #x0000000000000001 ) ) ( = z! #x0000000000000001 ) ) ( and ( not ( = w #x0000000000000001 ) ) ( = z #x0000000000000000 ) ( = x! x ) ( = w! w ) ( = y! ( bvadd y #x0000000000000001 ) ) ( = z! #x0000000000000001 ) ) ( and ( = w #x0000000000000001 ) ( not ( = z #x0000000000000000 ) ) ( = x! ( bvadd x #x0000000000000001 ) ) ( = w! #x0000000000000000 ) ( = y! y ) ( = z! z ) ) ( and ( not ( = w #x0000000000000001 ) ) ( not ( = z #x0000000000000000 ) ) ( = x! x ) ( = w! w ) ( = y! y ) ( = z! z ) ) ) ) ) ( and a!1 ) ))
(define-fun post_fun ((z (_ BitVec 64)) (w (_ BitVec 64)) (y (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( = x y ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

