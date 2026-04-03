(set-logic BV)

(synth-inv inv_fun ((x (_ BitVec 64))))

(define-fun pre_fun ((x (_ BitVec 64))) Bool
       ( and ( bvsle x ( bvadd #x0000000000000001 ( bvnot #x0000000000000002 ) ) ) ( bvsge x ( bvadd #x0000000000000001 ( bvnot #x0000000000000003 ) ) ) ))
(define-fun trans_fun ((x! (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( let ( ( a!1 ( or ( and ( = x! ( bvadd x #x0000000000000002 ) ) ( bvslt x #x0000000000000001 ) ) ( and ( = x! ( bvadd x #x0000000000000001 ) ) ( bvsge x #x0000000000000001 ) ) ) ) ) ( and a!1 ) ))
(define-fun post_fun ((x (_ BitVec 64))) Bool
       ( bvsle #xfffffffffffffffb x ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

