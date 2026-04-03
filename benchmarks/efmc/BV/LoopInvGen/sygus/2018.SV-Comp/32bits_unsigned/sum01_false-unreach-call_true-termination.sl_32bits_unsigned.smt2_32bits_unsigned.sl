(set-logic BV)

(synth-inv inv_fun ((sn (_ BitVec 32))(n (_ BitVec 32))(i (_ BitVec 32))))

(define-fun pre_fun ((sn (_ BitVec 32)) (n (_ BitVec 32)) (i (_ BitVec 32))) Bool
       ( and ( = i #x00000001 ) ( = sn #x00000000 ) ( bvuge n #x00000000 ) ))
(define-fun trans_fun ((sn! (_ BitVec 32)) (n! (_ BitVec 32)) (i! (_ BitVec 32)) (sn (_ BitVec 32)) (n (_ BitVec 32)) (i (_ BitVec 32))) Bool
       ( and ( bvule i n ) ( ite ( bvult i #x0000000a ) ( = sn! ( bvadd sn #x00000002 ) ) ( = sn! sn ) ) ( = i! ( bvadd i #x00000001 ) ) ( = n! n ) ))
(define-fun post_fun ((sn (_ BitVec 32)) (n (_ BitVec 32)) (i (_ BitVec 32))) Bool
       ( or ( = sn #x00000000 ) ( = sn ( bvmul #x00000002 n ) ) ( bvule i n ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

