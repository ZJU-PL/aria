(set-logic BV)

(synth-inv inv_fun ((y (_ BitVec 64))(x (_ BitVec 64))(n (_ BitVec 64))))

(define-fun pre_fun ((y (_ BitVec 64)) (x (_ BitVec 64)) (n (_ BitVec 64))) Bool
       ( and ( = n #x0000000000000000 ) ( bvuge x #x0000000000000000 ) ( bvuge y #x0000000000000000 ) ( = x y ) ))
(define-fun trans_fun ((y! (_ BitVec 64)) (x! (_ BitVec 64)) (n! (_ BitVec 64)) (y (_ BitVec 64)) (x (_ BitVec 64)) (n (_ BitVec 64))) Bool
       ( let ( ( a!1 ( or ( and ( = x n ) ( = n! n ) ( = x! x ) ( = y! y ) ) ( and ( not ( = x n ) ) ( = n! n ) ( = x! ( bvsub x #x0000000000000001 ) ) ( = y! ( bvsub y #x0000000000000001 ) ) ) ) ) ) ( and a!1 ) ))
(define-fun post_fun ((y (_ BitVec 64)) (x (_ BitVec 64)) (n (_ BitVec 64))) Bool
       ( or ( not ( = x n ) ) ( = y n ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

