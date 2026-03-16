(set-logic BV)

(synth-inv inv_fun ((y (_ BitVec 32))(x (_ BitVec 32))))

(define-fun pre_fun ((y (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( bvsle y #x000f4240 ))
(define-fun trans_fun ((y! (_ BitVec 32)) (x! (_ BitVec 32)) (y (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( and ( bvslt x #x00000064 ) ( bvsgt y #x00000000 ) ( = x! ( bvadd x y ) ) ( = y! y ) ))
(define-fun post_fun ((y (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( let ( ( a!1 ( or ( bvsle y #x00000000 ) ( and ( not ( bvsle y #x00000000 ) ) ( bvsle #x00000064 x ) ) ) ) ) ( or ( and ( not ( bvsle y #x00000000 ) ) ( bvsle #x00000064 x ) ) ( bvsle y #x00000000 ) ( not a!1 ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

