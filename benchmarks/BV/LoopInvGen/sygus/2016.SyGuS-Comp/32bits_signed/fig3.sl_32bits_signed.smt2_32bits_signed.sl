(set-logic BV)

(synth-inv inv_fun ((lock (_ BitVec 32))(y (_ BitVec 32))(x (_ BitVec 32))))

(define-fun pre_fun ((lock (_ BitVec 32)) (y (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( or ( and ( = x y ) ( = lock #x00000001 ) ) ( and ( = ( bvadd x #x00000001 ) y ) ( = lock #x00000000 ) ) ))
(define-fun trans_fun ((lock! (_ BitVec 32)) (y! (_ BitVec 32)) (x! (_ BitVec 32)) (lock (_ BitVec 32)) (y (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( let ( ( a!1 ( or ( and ( not ( = x y ) ) ( = lock! #x00000001 ) ( = x! y ) ) ( and ( not ( = x y ) ) ( = lock! #x00000000 ) ( = x! y ) ( = y! ( bvadd y #x00000001 ) ) ) ) ) ) ( and a!1 ) ))
(define-fun post_fun ((lock (_ BitVec 32)) (y (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( not ( and ( = x y ) ( not ( = lock #x00000001 ) ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

