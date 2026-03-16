(set-logic BV)

(synth-inv inv_fun ((y (_ BitVec 32))(x (_ BitVec 32))(j (_ BitVec 32))(i (_ BitVec 32))))

(define-fun pre_fun ((y (_ BitVec 32)) (x (_ BitVec 32)) (j (_ BitVec 32)) (i (_ BitVec 32))) Bool
       ( and ( bvuge i #x00000000 ) ( bvuge j #x00000000 ) ( = x i ) ( = y j ) ))
(define-fun trans_fun ((y! (_ BitVec 32)) (x! (_ BitVec 32)) (j! (_ BitVec 32)) (i! (_ BitVec 32)) (y (_ BitVec 32)) (x (_ BitVec 32)) (j (_ BitVec 32)) (i (_ BitVec 32))) Bool
       ( and ( not ( = x #x00000000 ) ) ( = i! i ) ( = j! j ) ( = x! ( bvsub x #x00000001 ) ) ( = y! ( bvsub y #x00000001 ) ) ))
(define-fun post_fun ((y (_ BitVec 32)) (x (_ BitVec 32)) (j (_ BitVec 32)) (i (_ BitVec 32))) Bool
       ( or ( not ( = i j ) ) ( = y #x00000000 ) ( not ( = x #x00000000 ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

