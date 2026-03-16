(set-logic BV)

(synth-inv inv_fun ((sn (_ BitVec 32))(i (_ BitVec 32))))

(define-fun pre_fun ((sn (_ BitVec 32)) (i (_ BitVec 32))) Bool
       ( and ( = sn #x00000000 ) ( = i #x00000001 ) ))
(define-fun trans_fun ((sn! (_ BitVec 32)) (i! (_ BitVec 32)) (sn (_ BitVec 32)) (i (_ BitVec 32))) Bool
       ( and ( bvule i #x00000008 ) ( ite ( bvult i #x00000004 ) ( = sn! ( bvadd sn #x00000002 ) ) ( = sn! sn ) ) ( = i! ( bvadd i #x00000001 ) ) ))
(define-fun post_fun ((sn (_ BitVec 32)) (i (_ BitVec 32))) Bool
       ( or ( = sn #x00000010 ) ( = sn #x00000000 ) ( and ( = ( ( _ extract (_ bv31 32) (_ bv4 32) ) i ) #x0000000 ) ( bvule ( ( _ extract (_ bv3 32) (_ bv0 32) ) i ) #x8 ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

