(set-logic BV)

(synth-inv inv_fun ((x (_ BitVec 64))(n1 (_ BitVec 64))(loop1 (_ BitVec 64))(sn (_ BitVec 64))))

(define-fun pre_fun ((x (_ BitVec 64)) (n1 (_ BitVec 64)) (loop1 (_ BitVec 64)) (sn (_ BitVec 64))) Bool
       ( and ( = sn #x0000000000000000 ) ( = x #x0000000000000000 ) ( bvuge loop1 #x0000000000000000 ) ( bvuge n1 #x0000000000000000 ) ))
(define-fun trans_fun ((x! (_ BitVec 64)) (n1! (_ BitVec 64)) (loop1! (_ BitVec 64)) (sn! (_ BitVec 64)) (x (_ BitVec 64)) (n1 (_ BitVec 64)) (loop1 (_ BitVec 64)) (sn (_ BitVec 64))) Bool
       ( and ( ite ( bvult x #x000000000000000a ) ( = sn! ( bvadd sn #x0000000000000002 ) ) ( = sn! sn ) ) ( = x! ( bvadd x #x0000000000000001 ) ) ( = loop1! loop1 ) ( = n1! n1 ) ))
(define-fun post_fun ((x (_ BitVec 64)) (n1 (_ BitVec 64)) (loop1 (_ BitVec 64)) (sn (_ BitVec 64))) Bool
       ( or ( = sn #x0000000000000000 ) ( = sn ( bvmul #x0000000000000002 x ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

