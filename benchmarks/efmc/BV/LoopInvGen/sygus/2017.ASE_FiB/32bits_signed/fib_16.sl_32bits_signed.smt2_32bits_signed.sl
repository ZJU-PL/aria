(set-logic BV)

(synth-inv inv_fun ((j (_ BitVec 32))(i (_ BitVec 32))(y (_ BitVec 32))(x (_ BitVec 32))))

(define-fun pre_fun ((j (_ BitVec 32)) (i (_ BitVec 32)) (y (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( and ( = x i ) ( = y j ) ))
(define-fun trans_fun ((j! (_ BitVec 32)) (i! (_ BitVec 32)) (y! (_ BitVec 32)) (x! (_ BitVec 32)) (j (_ BitVec 32)) (i (_ BitVec 32)) (y (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( let ( ( a!1 ( or ( and ( not ( = x #x00000000 ) ) ( = x! ( bvsub x #x00000001 ) ) ( = y! ( bvsub y #x00000001 ) ) ( = i! i ) ( = j! j ) ) ( and ( = x #x00000000 ) ( = x! x ) ( = y! y ) ( = i! i ) ( = j! j ) ) ) ) ) ( and a!1 ) ))
(define-fun post_fun ((j (_ BitVec 32)) (i (_ BitVec 32)) (y (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( or ( = y #x00000000 ) ( not ( = x #x00000000 ) ) ( not ( = i j ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

