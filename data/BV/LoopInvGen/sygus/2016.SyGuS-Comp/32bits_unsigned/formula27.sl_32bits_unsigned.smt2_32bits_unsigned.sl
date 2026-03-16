(set-logic BV)

(synth-inv inv_fun ((x5 (_ BitVec 32))(x4 (_ BitVec 32))(x3 (_ BitVec 32))(x2 (_ BitVec 32))(x1 (_ BitVec 32))))

(define-fun pre_fun ((x5 (_ BitVec 32)) (x4 (_ BitVec 32)) (x3 (_ BitVec 32)) (x2 (_ BitVec 32)) (x1 (_ BitVec 32))) Bool
       ( and ( = x1 #x00000000 ) ( = x2 #x00000000 ) ( = x3 #x00000000 ) ( = x4 #x00000000 ) ( = x5 #x00000000 ) ))
(define-fun trans_fun ((x5! (_ BitVec 32)) (x4! (_ BitVec 32)) (x3! (_ BitVec 32)) (x2! (_ BitVec 32)) (x1! (_ BitVec 32)) (x5 (_ BitVec 32)) (x4 (_ BitVec 32)) (x3 (_ BitVec 32)) (x2 (_ BitVec 32)) (x1 (_ BitVec 32))) Bool
       ( let ( ( a!1 ( or ( bvule x2! ( bvadd #x00000001 ( bvnot #x00000001 ) ) ) ( bvule x4! ( bvadd x2! #x00000002 ) ) ) ) ) ( and ( bvule #x00000000 x1! ) ( bvule x1! ( bvadd x4! #x00000001 ) ) ( = x2! x3! ) ( = #x00000000 x5! ) a!1 ) ))
(define-fun post_fun ((x5 (_ BitVec 32)) (x4 (_ BitVec 32)) (x3 (_ BitVec 32)) (x2 (_ BitVec 32)) (x1 (_ BitVec 32))) Bool
       ( and ( bvule x1 ( bvadd #x00000001 x4 ) ) ( = x2 x3 ) ( = x5 #x00000000 ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

