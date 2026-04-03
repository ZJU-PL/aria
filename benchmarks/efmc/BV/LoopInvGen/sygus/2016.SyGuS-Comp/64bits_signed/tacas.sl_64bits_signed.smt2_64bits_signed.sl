(set-logic BV)

(synth-inv inv_fun ((y (_ BitVec 64))(x (_ BitVec 64))(j (_ BitVec 64))(i (_ BitVec 64))))

(define-fun pre_fun ((y (_ BitVec 64)) (x (_ BitVec 64)) (j (_ BitVec 64)) (i (_ BitVec 64))) Bool
       ( and ( = i x ) ( = j y ) ))
(define-fun trans_fun ((y! (_ BitVec 64)) (x! (_ BitVec 64)) (j! (_ BitVec 64)) (i! (_ BitVec 64)) (y (_ BitVec 64)) (x (_ BitVec 64)) (j (_ BitVec 64)) (i (_ BitVec 64))) Bool
       ( and ( = i! i ) ( = j! j ) ( not ( = x #x0000000000000000 ) ) ( = x! ( bvsub x #x0000000000000001 ) ) ( = y! ( bvsub y #x0000000000000001 ) ) ))
(define-fun post_fun ((y (_ BitVec 64)) (x (_ BitVec 64)) (j (_ BitVec 64)) (i (_ BitVec 64))) Bool
       ( or ( not ( = x #x0000000000000000 ) ) ( = y #x0000000000000000 ) ( not ( = i j ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

