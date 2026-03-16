(set-logic BV)

(synth-inv inv_fun ((i (_ BitVec 64))(sum (_ BitVec 64))(n (_ BitVec 64))))

(define-fun pre_fun ((i (_ BitVec 64)) (sum (_ BitVec 64)) (n (_ BitVec 64))) Bool
       ( and ( bvsle #x0000000000000001 n ) ( bvsle n #x00000000000003e8 ) ( = sum #x0000000000000000 ) ( = i #x0000000000000001 ) ))
(define-fun trans_fun ((i! (_ BitVec 64)) (sum! (_ BitVec 64)) (n! (_ BitVec 64)) (i (_ BitVec 64)) (sum (_ BitVec 64)) (n (_ BitVec 64))) Bool
       ( and ( bvsle i n ) ( = i! ( bvadd i #x0000000000000001 ) ) ( = sum! ( bvadd sum i ) ) ( = n! n ) ))
(define-fun post_fun ((i (_ BitVec 64)) (sum (_ BitVec 64)) (n (_ BitVec 64))) Bool
       ( or ( bvsle i n ) ( = ( bvmul #x0000000000000002 sum ) ( bvmul n ( bvadd #x0000000000000001 n ) ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

