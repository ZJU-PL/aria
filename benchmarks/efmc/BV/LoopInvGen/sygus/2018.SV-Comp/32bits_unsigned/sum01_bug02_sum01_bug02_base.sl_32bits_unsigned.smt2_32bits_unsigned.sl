(set-logic BV)

(synth-inv inv_fun ((sn (_ BitVec 32))(n (_ BitVec 32))(i (_ BitVec 32))))

(define-fun pre_fun ((sn (_ BitVec 32)) (n (_ BitVec 32)) (i (_ BitVec 32))) Bool
       ( and ( = i #x00000001 ) ( = sn #x00000000 ) ( bvuge n #x00000000 ) ))
(define-fun trans_fun ((sn! (_ BitVec 32)) (n! (_ BitVec 32)) (i! (_ BitVec 32)) (sn (_ BitVec 32)) (n (_ BitVec 32)) (i (_ BitVec 32))) Bool
       ( and ( bvule i n ) ( ite ( = i #x00000004 ) ( = sn! ( bvsub #x00000000 #x0000000a ) ) ( = sn! ( bvadd sn #x00000002 ) ) ) ( = i! ( bvadd i #x00000001 ) ) ( = n! n ) ))
(define-fun post_fun ((sn (_ BitVec 32)) (n (_ BitVec 32)) (i (_ BitVec 32))) Bool
       ( or ( = sn ( bvmul #x00000002 n ) ) ( bvule i n ) ( = sn #x00000000 ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

