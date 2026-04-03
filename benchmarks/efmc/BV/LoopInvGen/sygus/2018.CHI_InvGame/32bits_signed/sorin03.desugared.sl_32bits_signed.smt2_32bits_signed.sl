(set-logic BV)

(synth-inv inv_fun ((j (_ BitVec 32))(i (_ BitVec 32))(s (_ BitVec 32))(n (_ BitVec 32))))

(define-fun pre_fun ((j (_ BitVec 32)) (i (_ BitVec 32)) (s (_ BitVec 32)) (n (_ BitVec 32))) Bool
       ( and ( bvsle #x00000001 n ) ( bvsle n #x000003e8 ) ( = s #x00000000 ) ( = j #x00000000 ) ( = i #x00000001 ) ))
(define-fun trans_fun ((j! (_ BitVec 32)) (i! (_ BitVec 32)) (s! (_ BitVec 32)) (n! (_ BitVec 32)) (j (_ BitVec 32)) (i (_ BitVec 32)) (s (_ BitVec 32)) (n (_ BitVec 32))) Bool
       ( and ( bvsle i n ) ( = s! ( bvadd s i ) ) ( = j! i ) ( = i! ( bvadd i #x00000001 ) ) ( = n! n ) ))
(define-fun post_fun ((j (_ BitVec 32)) (i (_ BitVec 32)) (s (_ BitVec 32)) (n (_ BitVec 32))) Bool
       ( or ( bvsle i n ) ( = ( bvmul #x00000002 s ) ( bvmul n ( bvadd #x00000001 n ) ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

