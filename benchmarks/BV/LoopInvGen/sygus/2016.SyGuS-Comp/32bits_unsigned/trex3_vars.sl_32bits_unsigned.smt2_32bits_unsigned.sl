(set-logic BV)

(synth-inv inv_fun ((v3 (_ BitVec 32))(v2 (_ BitVec 32))(v1 (_ BitVec 32))(d3 (_ BitVec 32))(d2 (_ BitVec 32))(d1 (_ BitVec 32))(x3 (_ BitVec 32))(x2 (_ BitVec 32))(x1 (_ BitVec 32))))

(define-fun pre_fun ((v3 (_ BitVec 32)) (v2 (_ BitVec 32)) (v1 (_ BitVec 32)) (d3 (_ BitVec 32)) (d2 (_ BitVec 32)) (d1 (_ BitVec 32)) (x3 (_ BitVec 32)) (x2 (_ BitVec 32)) (x1 (_ BitVec 32))) Bool
       ( and ( = d1 #x00000001 ) ( = d2 #x00000001 ) ( = d3 #x00000001 ) ))
(define-fun trans_fun ((v3! (_ BitVec 32)) (v2! (_ BitVec 32)) (v1! (_ BitVec 32)) (d3! (_ BitVec 32)) (d2! (_ BitVec 32)) (d1! (_ BitVec 32)) (x3! (_ BitVec 32)) (x2! (_ BitVec 32)) (x1! (_ BitVec 32)) (v3 (_ BitVec 32)) (v2 (_ BitVec 32)) (v1 (_ BitVec 32)) (d3 (_ BitVec 32)) (d2 (_ BitVec 32)) (d1 (_ BitVec 32)) (x3 (_ BitVec 32)) (x2 (_ BitVec 32)) (x1 (_ BitVec 32))) Bool
       ( let ( ( a!1 ( and ( = d1! d1 ) ( = d2! d2 ) ( = d3! d3 ) ) ) ) ( let ( ( a!2 ( and ( = x2! x2 ) ( and ( = x3! x3 ) a!1 ) ( bvugt x1 #x00000000 ) ( bvugt x2 #x00000000 ) ( bvugt x3 #x00000000 ) ( = x1! ( bvadd x1 ( bvsub #x00000000 d1 ) ) ) ) ) ( a!3 ( and ( = x1! x1 ) ( and ( = x3! x3 ) a!1 ) ( bvugt x1 #x00000000 ) ( bvugt x2 #x00000000 ) ( bvugt x3 #x00000000 ) ( = x2! ( bvadd x2 ( bvsub #x00000000 d2 ) ) ) ) ) ( a!4 ( and ( = x1! x1 ) ( = x2! x2 ) a!1 ( bvugt x1 #x00000000 ) ( bvugt x2 #x00000000 ) ( bvugt x3 #x00000000 ) ( = x3! ( bvadd x3 ( bvsub #x00000000 d3 ) ) ) ) ) ) ( and ( or a!2 a!3 a!4 ) ) ) ))
(define-fun post_fun ((v3 (_ BitVec 32)) (v2 (_ BitVec 32)) (v1 (_ BitVec 32)) (d3 (_ BitVec 32)) (d2 (_ BitVec 32)) (d1 (_ BitVec 32)) (x3 (_ BitVec 32)) (x2 (_ BitVec 32)) (x1 (_ BitVec 32))) Bool
       ( or ( = x1 #x00000000 ) ( = x2 #x00000000 ) ( = x3 #x00000000 ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

