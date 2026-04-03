(set-logic BV)

(synth-inv inv_fun ((z3 (_ BitVec 32))(z2 (_ BitVec 32))(z1 (_ BitVec 32))(y (_ BitVec 32))(x (_ BitVec 32))(j (_ BitVec 32))(i (_ BitVec 32))))

(define-fun pre_fun ((z3 (_ BitVec 32)) (z2 (_ BitVec 32)) (z1 (_ BitVec 32)) (y (_ BitVec 32)) (x (_ BitVec 32)) (j (_ BitVec 32)) (i (_ BitVec 32))) Bool
       ( and ( = i x ) ( = j y ) ))
(define-fun trans_fun ((z3! (_ BitVec 32)) (z2! (_ BitVec 32)) (z1! (_ BitVec 32)) (y! (_ BitVec 32)) (x! (_ BitVec 32)) (j! (_ BitVec 32)) (i! (_ BitVec 32)) (z3 (_ BitVec 32)) (z2 (_ BitVec 32)) (z1 (_ BitVec 32)) (y (_ BitVec 32)) (x (_ BitVec 32)) (j (_ BitVec 32)) (i (_ BitVec 32))) Bool
       ( and ( = i! i ) ( = j! j ) ( not ( = x #x00000000 ) ) ( = x! ( bvsub x #x00000001 ) ) ( = y! ( bvsub y #x00000001 ) ) ))
(define-fun post_fun ((z3 (_ BitVec 32)) (z2 (_ BitVec 32)) (z1 (_ BitVec 32)) (y (_ BitVec 32)) (x (_ BitVec 32)) (j (_ BitVec 32)) (i (_ BitVec 32))) Bool
       ( or ( not ( = x #x00000000 ) ) ( = y #x00000000 ) ( not ( = i j ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

