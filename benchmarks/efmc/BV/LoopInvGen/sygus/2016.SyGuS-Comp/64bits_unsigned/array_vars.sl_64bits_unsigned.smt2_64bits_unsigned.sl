(set-logic BV)

(synth-inv inv_fun ((size (_ BitVec 64))(v3 (_ BitVec 64))(v2 (_ BitVec 64))(v1 (_ BitVec 64))(z (_ BitVec 64))(y (_ BitVec 64))(x (_ BitVec 64))))

(define-fun pre_fun ((size (_ BitVec 64)) (v3 (_ BitVec 64)) (v2 (_ BitVec 64)) (v1 (_ BitVec 64)) (z (_ BitVec 64)) (y (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( = x #x0000000000000000 ))
(define-fun trans_fun ((size! (_ BitVec 64)) (v3! (_ BitVec 64)) (v2! (_ BitVec 64)) (v1! (_ BitVec 64)) (z! (_ BitVec 64)) (y! (_ BitVec 64)) (x! (_ BitVec 64)) (size (_ BitVec 64)) (v3 (_ BitVec 64)) (v2 (_ BitVec 64)) (v1 (_ BitVec 64)) (z (_ BitVec 64)) (y (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( let ( ( a!1 ( or ( and ( = x! ( bvadd x #x0000000000000001 ) ) ( = y! z! ) ( bvule z! y ) ( bvult x size ) ) ( and ( = x! ( bvadd x #x0000000000000001 ) ) ( = y! y ) ( bvugt z! y ) ( bvult x size ) ) ) ) ) ( and a!1 ) ))
(define-fun post_fun ((size (_ BitVec 64)) (v3 (_ BitVec 64)) (v2 (_ BitVec 64)) (v1 (_ BitVec 64)) (z (_ BitVec 64)) (y (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( not ( and ( bvule size x ) ( not ( bvule y z ) ) ( not ( = size #x0000000000000000 ) ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

