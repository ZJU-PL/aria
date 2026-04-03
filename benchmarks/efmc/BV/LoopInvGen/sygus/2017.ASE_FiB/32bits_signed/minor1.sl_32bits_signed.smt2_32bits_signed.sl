(set-logic BV)

(synth-inv inv_fun ((x (_ BitVec 32))))

(define-fun pre_fun ((x (_ BitVec 32))) Bool
       ( and ( bvsle x ( bvadd #x00000001 ( bvnot #x00000002 ) ) ) ( bvsge x ( bvadd #x00000001 ( bvnot #x00000003 ) ) ) ))
(define-fun trans_fun ((x! (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( let ( ( a!1 ( or ( and ( = x! ( bvadd x #x00000002 ) ) ( bvslt x #x00000001 ) ) ( and ( = x! ( bvadd x #x00000001 ) ) ( bvsge x #x00000001 ) ) ) ) ) ( and a!1 ) ))
(define-fun post_fun ((x (_ BitVec 32))) Bool
       ( bvsle #xfffffffb x ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

