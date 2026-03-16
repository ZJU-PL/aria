(set-logic BV)

(synth-inv inv_fun ((sn (_ BitVec 32))(n (_ BitVec 32))(i (_ BitVec 32))))

(define-fun pre_fun ((sn (_ BitVec 32)) (n (_ BitVec 32)) (i (_ BitVec 32))) Bool
       ( and ( = i #x00000001 ) ( = sn #x00000000 ) ( bvsge n #x00000000 ) ))
(define-fun trans_fun ((sn! (_ BitVec 32)) (n! (_ BitVec 32)) (i! (_ BitVec 32)) (sn (_ BitVec 32)) (n (_ BitVec 32)) (i (_ BitVec 32))) Bool
       ( and ( bvsle i n ) ( ite ( bvslt i #x0000000a ) ( = sn! ( bvadd sn #x00000002 ) ) ( = sn! sn ) ) ( = i! ( bvadd i #x00000001 ) ) ( = n! n ) ))
(define-fun post_fun ((sn (_ BitVec 32)) (n (_ BitVec 32)) (i (_ BitVec 32))) Bool
       ( or ( bvsle i n ) ( = sn #x00000000 ) ( = sn ( bvmul #x00000002 n ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

