(set-logic BV)

(synth-inv inv_fun ((j (_ BitVec 64))(i (_ BitVec 64))(s (_ BitVec 64))(n (_ BitVec 64))))

(define-fun pre_fun ((j (_ BitVec 64)) (i (_ BitVec 64)) (s (_ BitVec 64)) (n (_ BitVec 64))) Bool
       ( and ( bvsle #x0000000000000001 n ) ( bvsle n #x00000000000003e8 ) ( = s #x0000000000000000 ) ( = j #x0000000000000000 ) ( = i #x0000000000000001 ) ))
(define-fun trans_fun ((j! (_ BitVec 64)) (i! (_ BitVec 64)) (s! (_ BitVec 64)) (n! (_ BitVec 64)) (j (_ BitVec 64)) (i (_ BitVec 64)) (s (_ BitVec 64)) (n (_ BitVec 64))) Bool
       ( and ( bvsle i n ) ( = s! ( bvadd s i ) ) ( = j! i ) ( = i! ( bvadd i #x0000000000000001 ) ) ( = n! n ) ))
(define-fun post_fun ((j (_ BitVec 64)) (i (_ BitVec 64)) (s (_ BitVec 64)) (n (_ BitVec 64))) Bool
       ( or ( bvsle i n ) ( = ( bvmul #x0000000000000002 s ) ( bvmul n ( bvadd #x0000000000000001 n ) ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

