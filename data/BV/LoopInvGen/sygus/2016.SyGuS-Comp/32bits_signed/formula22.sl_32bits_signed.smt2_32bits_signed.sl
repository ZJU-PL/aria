(set-logic BV)

(synth-inv inv_fun ((x3 (_ BitVec 32))(x2 (_ BitVec 32))(x1 (_ BitVec 32))))

(define-fun pre_fun ((x3 (_ BitVec 32)) (x2 (_ BitVec 32)) (x1 (_ BitVec 32))) Bool
       ( and ( = x1 #x00000000 ) ( = x2 #x00000000 ) ( = x3 #x00000000 ) ))
(define-fun trans_fun ((x3! (_ BitVec 32)) (x2! (_ BitVec 32)) (x1! (_ BitVec 32)) (x3 (_ BitVec 32)) (x2 (_ BitVec 32)) (x1 (_ BitVec 32))) Bool
       ( and ( bvsle x1! x2! ) ( or ( bvsge x2! #x00000000 ) ( bvsle ( bvsub x2! x3! ) #x00000002 ) ) ))
(define-fun post_fun ((x3 (_ BitVec 32)) (x2 (_ BitVec 32)) (x1 (_ BitVec 32))) Bool
       ( let ( ( a!1 ( or ( bvsle #x00000000 x2 ) ( bvsle ( bvadd x2 ( bvmul #xffffffff x3 ) ) #x00000002 ) ) ) ) ( and ( bvsle x1 x2 ) a!1 ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

