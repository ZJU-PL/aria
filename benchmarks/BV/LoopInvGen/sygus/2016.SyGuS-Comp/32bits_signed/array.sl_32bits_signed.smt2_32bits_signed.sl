(set-logic BV)

(synth-inv inv_fun ((z (_ BitVec 32))(y (_ BitVec 32))(x (_ BitVec 32))))

(define-fun pre_fun ((z (_ BitVec 32)) (y (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( = x #x00000000 ))
(define-fun trans_fun ((z! (_ BitVec 32)) (y! (_ BitVec 32)) (x! (_ BitVec 32)) (z (_ BitVec 32)) (y (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( let ( ( a!1 ( or ( and ( = x! ( bvadd x #x00000001 ) ) ( = y! z! ) ( bvsle z! y ) ( bvslt x #x00000005 ) ) ( and ( = x! ( bvadd x #x00000001 ) ) ( = y! y ) ( bvsgt z! y ) ( bvslt x #x00000005 ) ) ) ) ) ( and a!1 ) ))
(define-fun post_fun ((z (_ BitVec 32)) (y (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( not ( and ( bvsle #x00000005 x ) ( not ( bvsle y z ) ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

