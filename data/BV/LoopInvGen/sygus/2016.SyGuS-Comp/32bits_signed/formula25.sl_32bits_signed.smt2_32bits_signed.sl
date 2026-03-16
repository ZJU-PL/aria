(set-logic BV)

(synth-inv inv_fun ((x4 (_ BitVec 32))(x3 (_ BitVec 32))(x2 (_ BitVec 32))(x1 (_ BitVec 32))))

(define-fun pre_fun ((x4 (_ BitVec 32)) (x3 (_ BitVec 32)) (x2 (_ BitVec 32)) (x1 (_ BitVec 32))) Bool
       ( and ( = x1 #x00000000 ) ( = x2 #x00000000 ) ( = x3 #x00000000 ) ( = x4 ( bvadd #x00000001 ( bvnot #x00000001 ) ) ) ))
(define-fun trans_fun ((x4! (_ BitVec 32)) (x3! (_ BitVec 32)) (x2! (_ BitVec 32)) (x1! (_ BitVec 32)) (x4 (_ BitVec 32)) (x3 (_ BitVec 32)) (x2 (_ BitVec 32)) (x1 (_ BitVec 32))) Bool
       ( and ( bvsle x1! #x00000000 ) ( bvsge x1! ( bvadd x4! #x00000001 ) ) ( = x2! x3! ) ( or ( bvsge x4! #x00000000 ) ( bvsle x4! x3! ) ) ))
(define-fun post_fun ((x4 (_ BitVec 32)) (x3 (_ BitVec 32)) (x2 (_ BitVec 32)) (x1 (_ BitVec 32))) Bool
       ( and ( bvsle x1 #x00000000 ) ( bvsle ( bvadd #x00000001 x4 ) x1 ) ( = x2 x3 ) ( or ( bvsle #x00000000 x4 ) ( bvsle x4 x3 ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

