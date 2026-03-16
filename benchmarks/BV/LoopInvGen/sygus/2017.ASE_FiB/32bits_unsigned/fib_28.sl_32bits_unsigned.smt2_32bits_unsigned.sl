(set-logic BV)

(synth-inv inv_fun ((y (_ BitVec 32))(x (_ BitVec 32))(n (_ BitVec 32))))

(define-fun pre_fun ((y (_ BitVec 32)) (x (_ BitVec 32)) (n (_ BitVec 32))) Bool
       ( and ( = n #x00000000 ) ( bvuge x #x00000000 ) ( bvuge y #x00000000 ) ( = x y ) ))
(define-fun trans_fun ((y! (_ BitVec 32)) (x! (_ BitVec 32)) (n! (_ BitVec 32)) (y (_ BitVec 32)) (x (_ BitVec 32)) (n (_ BitVec 32))) Bool
       ( let ( ( a!1 ( or ( and ( = x n ) ( = n! n ) ( = x! x ) ( = y! y ) ) ( and ( not ( = x n ) ) ( = n! n ) ( = x! ( bvsub x #x00000001 ) ) ( = y! ( bvsub y #x00000001 ) ) ) ) ) ) ( and a!1 ) ))
(define-fun post_fun ((y (_ BitVec 32)) (x (_ BitVec 32)) (n (_ BitVec 32))) Bool
       ( or ( not ( = x n ) ) ( = y n ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

