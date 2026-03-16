(set-logic BV)

(synth-inv inv_fun ((y (_ BitVec 64))(x (_ BitVec 64))))

(define-fun pre_fun ((y (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( bvsle y #x00000000000f4240 ))
(define-fun trans_fun ((y! (_ BitVec 64)) (x! (_ BitVec 64)) (y (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( and ( bvslt x #x0000000000000064 ) ( bvsgt y #x0000000000000000 ) ( = x! ( bvadd x y ) ) ( = y! y ) ))
(define-fun post_fun ((y (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( let ( ( a!1 ( or ( bvsle y #x0000000000000000 ) ( and ( not ( bvsle y #x0000000000000000 ) ) ( bvsle #x0000000000000064 x ) ) ) ) ) ( or ( not a!1 ) ( bvsle y #x0000000000000000 ) ( and ( not ( bvsle y #x0000000000000000 ) ) ( bvsle #x0000000000000064 x ) ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

