(set-logic BV)

(synth-inv inv_fun ((x5 (_ BitVec 64))(x4 (_ BitVec 64))(x3 (_ BitVec 64))(x2 (_ BitVec 64))(x1 (_ BitVec 64))))

(define-fun pre_fun ((x5 (_ BitVec 64)) (x4 (_ BitVec 64)) (x3 (_ BitVec 64)) (x2 (_ BitVec 64)) (x1 (_ BitVec 64))) Bool
       ( and ( = x1 #x0000000000000000 ) ( = x2 #x0000000000000000 ) ( = x3 #x0000000000000000 ) ( = x4 #x0000000000000000 ) ( = x5 #x0000000000000000 ) ))
(define-fun trans_fun ((x5! (_ BitVec 64)) (x4! (_ BitVec 64)) (x3! (_ BitVec 64)) (x2! (_ BitVec 64)) (x1! (_ BitVec 64)) (x5 (_ BitVec 64)) (x4 (_ BitVec 64)) (x3 (_ BitVec 64)) (x2 (_ BitVec 64)) (x1 (_ BitVec 64))) Bool
       ( let ( ( a!1 ( or ( bvule x2! ( bvadd #x0000000000000001 ( bvnot #x0000000000000001 ) ) ) ( bvule x4! ( bvadd x2! #x0000000000000002 ) ) ) ) ) ( and ( bvule #x0000000000000000 x1! ) ( bvule x1! ( bvadd x4! #x0000000000000001 ) ) ( = x2! x3! ) ( = #x0000000000000000 x5! ) a!1 ) ))
(define-fun post_fun ((x5 (_ BitVec 64)) (x4 (_ BitVec 64)) (x3 (_ BitVec 64)) (x2 (_ BitVec 64)) (x1 (_ BitVec 64))) Bool
       ( and ( bvule x1 ( bvadd #x0000000000000001 x4 ) ) ( = x2 x3 ) ( = x5 #x0000000000000000 ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

