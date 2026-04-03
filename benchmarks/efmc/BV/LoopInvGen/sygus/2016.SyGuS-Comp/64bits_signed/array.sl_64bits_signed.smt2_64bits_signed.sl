(set-logic BV)

(synth-inv inv_fun ((z (_ BitVec 64))(y (_ BitVec 64))(x (_ BitVec 64))))

(define-fun pre_fun ((z (_ BitVec 64)) (y (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( = x #x0000000000000000 ))
(define-fun trans_fun ((z! (_ BitVec 64)) (y! (_ BitVec 64)) (x! (_ BitVec 64)) (z (_ BitVec 64)) (y (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( let ( ( a!1 ( or ( and ( = x! ( bvadd x #x0000000000000001 ) ) ( = y! z! ) ( bvsle z! y ) ( bvslt x #x0000000000000005 ) ) ( and ( = x! ( bvadd x #x0000000000000001 ) ) ( = y! y ) ( bvsgt z! y ) ( bvslt x #x0000000000000005 ) ) ) ) ) ( and a!1 ) ))
(define-fun post_fun ((z (_ BitVec 64)) (y (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( not ( and ( bvsle #x0000000000000005 x ) ( not ( bvsle y z ) ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

