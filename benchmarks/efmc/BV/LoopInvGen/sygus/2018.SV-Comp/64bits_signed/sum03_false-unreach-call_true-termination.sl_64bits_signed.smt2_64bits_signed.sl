(set-logic BV)

(synth-inv inv_fun ((x (_ BitVec 64))(n1 (_ BitVec 64))(loop1 (_ BitVec 64))(sn (_ BitVec 64))))

(define-fun pre_fun ((x (_ BitVec 64)) (n1 (_ BitVec 64)) (loop1 (_ BitVec 64)) (sn (_ BitVec 64))) Bool
       ( and ( = sn #x0000000000000000 ) ( = x #x0000000000000000 ) ( bvsge loop1 #x0000000000000000 ) ( bvsge n1 #x0000000000000000 ) ))
(define-fun trans_fun ((x! (_ BitVec 64)) (n1! (_ BitVec 64)) (loop1! (_ BitVec 64)) (sn! (_ BitVec 64)) (x (_ BitVec 64)) (n1 (_ BitVec 64)) (loop1 (_ BitVec 64)) (sn (_ BitVec 64))) Bool
       ( and ( ite ( bvslt x #x000000000000000a ) ( = sn! ( bvadd sn #x0000000000000002 ) ) ( = sn! sn ) ) ( = x! ( bvadd x #x0000000000000001 ) ) ( = loop1! loop1 ) ( = n1! n1 ) ))
(define-fun post_fun ((x (_ BitVec 64)) (n1 (_ BitVec 64)) (loop1 (_ BitVec 64)) (sn (_ BitVec 64))) Bool
       ( or ( = sn ( bvmul #x0000000000000002 x ) ) ( = sn #x0000000000000000 ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

