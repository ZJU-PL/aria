(set-logic BV)

(synth-inv inv_fun ((y (_ BitVec 32))(x (_ BitVec 32))))

(define-fun pre_fun ((y (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( and ( bvule x #x00000001 ) ( bvuge x #x00000000 ) ( = y ( bvadd #x00000001 ( bvnot #x00000003 ) ) ) ))
(define-fun trans_fun ((y! (_ BitVec 32)) (x! (_ BitVec 32)) (y (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( let ( ( a!1 ( or ( and ( = x! ( bvsub x #x00000001 ) ) ( = y! ( bvadd y #x00000002 ) ) ( bvult ( bvsub x y ) #x00000002 ) ) ( and ( = x! x ) ( = y! ( bvadd y #x00000001 ) ) ( bvuge ( bvsub x y ) #x00000002 ) ) ) ) ) ( and a!1 ) ))
(define-fun post_fun ((y (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( and ( = ( ( _ extract (_ bv31 32) (_ bv1 32) ) x ) #b0000000000000000000000000000000 ) ( bvule #xfffffffd y ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

