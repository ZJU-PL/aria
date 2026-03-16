(set-logic BV)

(synth-inv inv_fun ((v3 (_ BitVec 64))(v2 (_ BitVec 64))(v1 (_ BitVec 64))(d3 (_ BitVec 64))(d2 (_ BitVec 64))(d1 (_ BitVec 64))(x3 (_ BitVec 64))(x2 (_ BitVec 64))(x1 (_ BitVec 64))))

(define-fun pre_fun ((v3 (_ BitVec 64)) (v2 (_ BitVec 64)) (v1 (_ BitVec 64)) (d3 (_ BitVec 64)) (d2 (_ BitVec 64)) (d1 (_ BitVec 64)) (x3 (_ BitVec 64)) (x2 (_ BitVec 64)) (x1 (_ BitVec 64))) Bool
       ( and ( = d1 #x0000000000000001 ) ( = d2 #x0000000000000001 ) ( = d3 #x0000000000000001 ) ))
(define-fun trans_fun ((v3! (_ BitVec 64)) (v2! (_ BitVec 64)) (v1! (_ BitVec 64)) (d3! (_ BitVec 64)) (d2! (_ BitVec 64)) (d1! (_ BitVec 64)) (x3! (_ BitVec 64)) (x2! (_ BitVec 64)) (x1! (_ BitVec 64)) (v3 (_ BitVec 64)) (v2 (_ BitVec 64)) (v1 (_ BitVec 64)) (d3 (_ BitVec 64)) (d2 (_ BitVec 64)) (d1 (_ BitVec 64)) (x3 (_ BitVec 64)) (x2 (_ BitVec 64)) (x1 (_ BitVec 64))) Bool
       ( let ( ( a!1 ( and ( = d1! d1 ) ( = d2! d2 ) ( = d3! d3 ) ) ) ) ( let ( ( a!2 ( and ( = x2! x2 ) ( and ( = x3! x3 ) a!1 ) ( bvugt x1 #x0000000000000000 ) ( bvugt x2 #x0000000000000000 ) ( bvugt x3 #x0000000000000000 ) ( = x1! ( bvadd x1 ( bvsub #x0000000000000000 d1 ) ) ) ) ) ( a!3 ( and ( = x1! x1 ) ( and ( = x3! x3 ) a!1 ) ( bvugt x1 #x0000000000000000 ) ( bvugt x2 #x0000000000000000 ) ( bvugt x3 #x0000000000000000 ) ( = x2! ( bvadd x2 ( bvsub #x0000000000000000 d2 ) ) ) ) ) ( a!4 ( and ( = x1! x1 ) ( = x2! x2 ) a!1 ( bvugt x1 #x0000000000000000 ) ( bvugt x2 #x0000000000000000 ) ( bvugt x3 #x0000000000000000 ) ( = x3! ( bvadd x3 ( bvsub #x0000000000000000 d3 ) ) ) ) ) ) ( and ( or a!2 a!3 a!4 ) ) ) ))
(define-fun post_fun ((v3 (_ BitVec 64)) (v2 (_ BitVec 64)) (v1 (_ BitVec 64)) (d3 (_ BitVec 64)) (d2 (_ BitVec 64)) (d1 (_ BitVec 64)) (x3 (_ BitVec 64)) (x2 (_ BitVec 64)) (x1 (_ BitVec 64))) Bool
       ( or ( = x1 #x0000000000000000 ) ( = x2 #x0000000000000000 ) ( = x3 #x0000000000000000 ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

