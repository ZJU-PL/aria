(set-logic BV)

(synth-inv inv_fun ((t (_ BitVec 64))(su (_ BitVec 64))(a (_ BitVec 64))(n (_ BitVec 64))))

(define-fun pre_fun ((t (_ BitVec 64)) (su (_ BitVec 64)) (a (_ BitVec 64)) (n (_ BitVec 64))) Bool
       ( and ( = a #x0000000000000000 ) ( = su #x0000000000000001 ) ( = t #x0000000000000001 ) ( bvsgt n #x0000000000000000 ) ))
(define-fun trans_fun ((t! (_ BitVec 64)) (su! (_ BitVec 64)) (a! (_ BitVec 64)) (n! (_ BitVec 64)) (t (_ BitVec 64)) (su (_ BitVec 64)) (a (_ BitVec 64)) (n (_ BitVec 64))) Bool
       ( and ( bvsle su n ) ( = n! n ) ( = a! ( bvadd a #x0000000000000001 ) ) ( = t! ( bvadd t #x0000000000000002 ) ) ( = su! ( bvadd su t! ) ) ))
(define-fun post_fun ((t (_ BitVec 64)) (su (_ BitVec 64)) (a (_ BitVec 64)) (n (_ BitVec 64))) Bool
       ( or ( bvsle su n ) ( = su ( bvmul ( bvadd #x0000000000000001 a ) ( bvadd #x0000000000000001 a ) ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

