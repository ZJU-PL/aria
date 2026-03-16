(set-logic BV)

(synth-inv inv_fun ((v3 (_ BitVec 64))(v2 (_ BitVec 64))(v1 (_ BitVec 64))(lock (_ BitVec 64))(y (_ BitVec 64))(x (_ BitVec 64))))

(define-fun pre_fun ((v3 (_ BitVec 64)) (v2 (_ BitVec 64)) (v1 (_ BitVec 64)) (lock (_ BitVec 64)) (y (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( or ( and ( = x y ) ( = lock #x0000000000000001 ) ) ( and ( = ( bvadd x #x0000000000000001 ) y ) ( = lock #x0000000000000000 ) ) ))
(define-fun trans_fun ((v3! (_ BitVec 64)) (v2! (_ BitVec 64)) (v1! (_ BitVec 64)) (lock! (_ BitVec 64)) (y! (_ BitVec 64)) (x! (_ BitVec 64)) (v3 (_ BitVec 64)) (v2 (_ BitVec 64)) (v1 (_ BitVec 64)) (lock (_ BitVec 64)) (y (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( let ( ( a!1 ( or ( and ( not ( = x y ) ) ( = lock! #x0000000000000001 ) ( = x! y ) ) ( and ( not ( = x y ) ) ( = lock! #x0000000000000000 ) ( = x! y ) ( = y! ( bvadd y #x0000000000000001 ) ) ) ) ) ) ( and a!1 ) ))
(define-fun post_fun ((v3 (_ BitVec 64)) (v2 (_ BitVec 64)) (v1 (_ BitVec 64)) (lock (_ BitVec 64)) (y (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( not ( and ( = x y ) ( not ( = lock #x0000000000000001 ) ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

