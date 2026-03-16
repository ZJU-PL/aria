(set-logic BV)

(synth-inv inv_fun ((j (_ BitVec 64))(i (_ BitVec 64))(y (_ BitVec 64))(x (_ BitVec 64))))

(define-fun pre_fun ((j (_ BitVec 64)) (i (_ BitVec 64)) (y (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( and ( = x i ) ( = y j ) ))
(define-fun trans_fun ((j! (_ BitVec 64)) (i! (_ BitVec 64)) (y! (_ BitVec 64)) (x! (_ BitVec 64)) (j (_ BitVec 64)) (i (_ BitVec 64)) (y (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( let ( ( a!1 ( or ( and ( not ( = x #x0000000000000000 ) ) ( = x! ( bvsub x #x0000000000000001 ) ) ( = y! ( bvsub y #x0000000000000001 ) ) ( = i! i ) ( = j! j ) ) ( and ( = x #x0000000000000000 ) ( = x! x ) ( = y! y ) ( = i! i ) ( = j! j ) ) ) ) ) ( and a!1 ) ))
(define-fun post_fun ((j (_ BitVec 64)) (i (_ BitVec 64)) (y (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( or ( not ( = x #x0000000000000000 ) ) ( not ( = i j ) ) ( = y #x0000000000000000 ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

