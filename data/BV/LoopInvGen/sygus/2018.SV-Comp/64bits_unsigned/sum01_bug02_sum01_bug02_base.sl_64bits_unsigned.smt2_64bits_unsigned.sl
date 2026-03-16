(set-logic BV)

(synth-inv inv_fun ((sn (_ BitVec 64))(n (_ BitVec 64))(i (_ BitVec 64))))

(define-fun pre_fun ((sn (_ BitVec 64)) (n (_ BitVec 64)) (i (_ BitVec 64))) Bool
       ( and ( = i #x0000000000000001 ) ( = sn #x0000000000000000 ) ( bvuge n #x0000000000000000 ) ))
(define-fun trans_fun ((sn! (_ BitVec 64)) (n! (_ BitVec 64)) (i! (_ BitVec 64)) (sn (_ BitVec 64)) (n (_ BitVec 64)) (i (_ BitVec 64))) Bool
       ( and ( bvule i n ) ( ite ( = i #x0000000000000004 ) ( = sn! ( bvsub #x0000000000000000 #x000000000000000a ) ) ( = sn! ( bvadd sn #x0000000000000002 ) ) ) ( = i! ( bvadd i #x0000000000000001 ) ) ( = n! n ) ))
(define-fun post_fun ((sn (_ BitVec 64)) (n (_ BitVec 64)) (i (_ BitVec 64))) Bool
       ( or ( = sn ( bvmul #x0000000000000002 n ) ) ( bvule i n ) ( = sn #x0000000000000000 ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

