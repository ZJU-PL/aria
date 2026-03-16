(set-logic BV)

(synth-inv inv_fun ((y (_ BitVec 64))(x (_ BitVec 64))))

(define-fun pre_fun ((y (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( and ( bvsle x #x0000000000000005 ) ( bvsge x #x0000000000000004 ) ( bvsle y #x0000000000000005 ) ( bvsge y #x0000000000000004 ) ))
(define-fun trans_fun ((y! (_ BitVec 64)) (x! (_ BitVec 64)) (y (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( let ( ( a!1 ( or ( and ( = x! ( bvsub x #x0000000000000001 ) ) ( = y! y ) ( bvsge x #x0000000000000000 ) ( bvsge y #x0000000000000000 ) ) ( and ( = x! x ) ( = y! ( bvsub y #x0000000000000001 ) ) ( bvslt x #x0000000000000000 ) ( bvsge y #x0000000000000000 ) ) ( and ( = x! ( bvadd x #x0000000000000001 ) ) ( = y! ( bvsub y #x0000000000000001 ) ) ( bvslt y #x0000000000000000 ) ) ) ) ) ( and a!1 ) ))
(define-fun post_fun ((y (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( bvsle y #x0000000000000005 ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

