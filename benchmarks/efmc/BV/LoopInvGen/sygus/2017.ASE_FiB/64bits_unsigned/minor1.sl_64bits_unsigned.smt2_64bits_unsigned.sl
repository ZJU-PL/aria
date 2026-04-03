(set-logic BV)

(synth-inv inv_fun ((x (_ BitVec 64))))

(define-fun pre_fun ((x (_ BitVec 64))) Bool
       ( and ( bvule x ( bvadd #x0000000000000001 ( bvnot #x0000000000000002 ) ) ) ( bvuge x ( bvadd #x0000000000000001 ( bvnot #x0000000000000003 ) ) ) ))
(define-fun trans_fun ((x! (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( let ( ( a!1 ( or ( and ( = x! ( bvadd x #x0000000000000002 ) ) ( bvult x #x0000000000000001 ) ) ( and ( = x! ( bvadd x #x0000000000000001 ) ) ( bvuge x #x0000000000000001 ) ) ) ) ) ( and a!1 ) ))
(define-fun post_fun ((x (_ BitVec 64))) Bool
       ( bvule #xfffffffffffffffb x ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

