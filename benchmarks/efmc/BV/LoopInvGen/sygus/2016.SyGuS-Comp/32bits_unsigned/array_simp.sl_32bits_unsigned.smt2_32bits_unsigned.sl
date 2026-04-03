(set-logic BV)

(synth-inv inv_fun ((size (_ BitVec 32))(z (_ BitVec 32))(y (_ BitVec 32))(x (_ BitVec 32))))

(define-fun pre_fun ((size (_ BitVec 32)) (z (_ BitVec 32)) (y (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( = x #x00000000 ))
(define-fun trans_fun ((size! (_ BitVec 32)) (z! (_ BitVec 32)) (y! (_ BitVec 32)) (x! (_ BitVec 32)) (size (_ BitVec 32)) (z (_ BitVec 32)) (y (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( let ( ( a!1 ( or ( and ( = x! ( bvadd x #x00000001 ) ) ( = y! z! ) ( bvule z! y ) ( bvult x size ) ) ( and ( = x! ( bvadd x #x00000001 ) ) ( = y! y ) ( bvugt z! y ) ( bvult x size ) ) ) ) ) ( and a!1 ) ))
(define-fun post_fun ((size (_ BitVec 32)) (z (_ BitVec 32)) (y (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( not ( and ( bvule size x ) ( not ( bvule y z ) ) ( not ( = size #x00000000 ) ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

