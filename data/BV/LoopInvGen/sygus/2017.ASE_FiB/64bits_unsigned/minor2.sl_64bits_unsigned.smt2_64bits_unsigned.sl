(set-logic BV)

(synth-inv inv_fun ((y (_ BitVec 64))(x (_ BitVec 64))))

(define-fun pre_fun ((y (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( and ( bvule x #x0000000000000001 ) ( bvuge x #x0000000000000000 ) ( = y ( bvadd #x0000000000000001 ( bvnot #x0000000000000003 ) ) ) ))
(define-fun trans_fun ((y! (_ BitVec 64)) (x! (_ BitVec 64)) (y (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( let ( ( a!1 ( or ( and ( = x! ( bvsub x #x0000000000000001 ) ) ( = y! ( bvadd y #x0000000000000002 ) ) ( bvult ( bvsub x y ) #x0000000000000002 ) ) ( and ( = x! x ) ( = y! ( bvadd y #x0000000000000001 ) ) ( bvuge ( bvsub x y ) #x0000000000000002 ) ) ) ) ) ( and a!1 ) ))
(define-fun post_fun ((y (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( and ( = ( ( _ extract (_ bv63 64) (_ bv1 64) ) x ) #b000000000000000000000000000000000000000000000000000000000000000 ) ( bvule #xfffffffffffffffd y ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

