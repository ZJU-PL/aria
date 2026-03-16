(set-logic BV)

(synth-inv inv_fun ((y (_ BitVec 32))(x (_ BitVec 32))))

(define-fun pre_fun ((y (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( and ( bvsle x #x00000005 ) ( bvsge x #x00000004 ) ( bvsle y #x00000005 ) ( bvsge y #x00000004 ) ))
(define-fun trans_fun ((y! (_ BitVec 32)) (x! (_ BitVec 32)) (y (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( let ( ( a!1 ( or ( and ( = x! ( bvsub x #x00000001 ) ) ( = y! y ) ( bvsge x #x00000000 ) ( bvsge y #x00000000 ) ) ( and ( = x! x ) ( = y! ( bvsub y #x00000001 ) ) ( bvslt x #x00000000 ) ( bvsge y #x00000000 ) ) ( and ( = x! ( bvadd x #x00000001 ) ) ( = y! ( bvsub y #x00000001 ) ) ( bvslt y #x00000000 ) ) ) ) ) ( and a!1 ) ))
(define-fun post_fun ((y (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( bvsle y #x00000005 ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

