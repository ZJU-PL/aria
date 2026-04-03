(set-logic BV)

(synth-inv inv_fun ((x (_ BitVec 32))))

(define-fun pre_fun ((x (_ BitVec 32))) Bool
       ( = x #x00000000 ))
(define-fun trans_fun ((x (_ BitVec 32)) (x! (_ BitVec 32))) Bool
       ( let ( ( a!1 ( or ( and ( bvslt x #x0000fff0 ) ( = x! ( bvadd x #x00000001 ) ) ) ( = x! ( bvadd x #x00000002 ) ) ) ) ) ( and ( bvslt x #x0fffffff ) a!1 ) ))
(define-fun post_fun ((x (_ BitVec 32))) Bool
       ( or ( = #x00000000 ( bvurem x #x00000002 ) ) ( bvslt x #x0fffffff ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

