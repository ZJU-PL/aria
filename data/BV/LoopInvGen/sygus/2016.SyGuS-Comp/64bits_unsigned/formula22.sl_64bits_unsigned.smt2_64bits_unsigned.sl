(set-logic BV)

(synth-inv inv_fun ((x3 (_ BitVec 64))(x2 (_ BitVec 64))(x1 (_ BitVec 64))))

(define-fun pre_fun ((x3 (_ BitVec 64)) (x2 (_ BitVec 64)) (x1 (_ BitVec 64))) Bool
       ( and ( = x1 #x0000000000000000 ) ( = x2 #x0000000000000000 ) ( = x3 #x0000000000000000 ) ))
(define-fun trans_fun ((x3! (_ BitVec 64)) (x2! (_ BitVec 64)) (x1! (_ BitVec 64)) (x3 (_ BitVec 64)) (x2 (_ BitVec 64)) (x1 (_ BitVec 64))) Bool
       ( and ( bvule x1! x2! ) ( or ( bvuge x2! #x0000000000000000 ) ( bvule ( bvsub x2! x3! ) #x0000000000000002 ) ) ))
(define-fun post_fun ((x3 (_ BitVec 64)) (x2 (_ BitVec 64)) (x1 (_ BitVec 64))) Bool
       ( bvule x1 x2 ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

