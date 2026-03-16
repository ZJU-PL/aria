(set-logic BV)

(synth-inv inv_fun ((y (_ BitVec 32))(x (_ BitVec 32))))

(define-fun pre_fun ((y (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( and ( = x #x00000000 ) ( = y #x00000000 ) ))
(define-fun trans_fun ((y! (_ BitVec 32)) (x! (_ BitVec 32)) (y (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( let ( ( a!1 ( or ( and ( = x! ( bvadd x #x00000001 ) ) ( = y! ( bvadd y #x00000064 ) ) ) ( and ( bvsge x #x00000004 ) ( = x! ( bvadd x #x00000001 ) ) ( = y! ( bvadd y #x00000001 ) ) ) ( and ( bvslt x #x00000000 ) ( = x! x ) ( = y! ( bvsub y #x00000001 ) ) ) ( and ( = x! x ) ( = y! y ) ) ) ) ) ( and a!1 ) ))
(define-fun post_fun ((y (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( or ( not ( bvsle #x00000004 x ) ) ( not ( bvsle y #x00000002 ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

