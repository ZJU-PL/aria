(set-logic BV)

(synth-inv inv_fun ((x4 (_ BitVec 64))(x3 (_ BitVec 64))(x2 (_ BitVec 64))(x1 (_ BitVec 64))))

(define-fun pre_fun ((x4 (_ BitVec 64)) (x3 (_ BitVec 64)) (x2 (_ BitVec 64)) (x1 (_ BitVec 64))) Bool
       ( and ( = x1 #x0000000000000000 ) ( = x2 #x0000000000000000 ) ( = x3 #x0000000000000000 ) ( = x4 ( bvadd #x0000000000000001 ( bvnot #x0000000000000001 ) ) ) ))
(define-fun trans_fun ((x4! (_ BitVec 64)) (x3! (_ BitVec 64)) (x2! (_ BitVec 64)) (x1! (_ BitVec 64)) (x4 (_ BitVec 64)) (x3 (_ BitVec 64)) (x2 (_ BitVec 64)) (x1 (_ BitVec 64))) Bool
       ( and ( bvule x1! #x0000000000000000 ) ( bvuge x1! ( bvadd x4! #x0000000000000001 ) ) ( = x2! x3! ) ( or ( bvuge x4! #x0000000000000000 ) ( bvule x4! x3! ) ) ))
(define-fun post_fun ((x4 (_ BitVec 64)) (x3 (_ BitVec 64)) (x2 (_ BitVec 64)) (x1 (_ BitVec 64))) Bool
       ( and ( = x1 #x0000000000000000 ) ( bvule ( bvadd #x0000000000000001 x4 ) x1 ) ( = x2 x3 ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

