(set-logic BV)

(synth-inv inv_fun ((x (_ BitVec 32))(n1 (_ BitVec 32))(loop1 (_ BitVec 32))(sn (_ BitVec 32))))

(define-fun pre_fun ((x (_ BitVec 32)) (n1 (_ BitVec 32)) (loop1 (_ BitVec 32)) (sn (_ BitVec 32))) Bool
       ( and ( = sn #x00000000 ) ( = x #x00000000 ) ( bvsge loop1 #x00000000 ) ( bvsge n1 #x00000000 ) ))
(define-fun trans_fun ((x! (_ BitVec 32)) (n1! (_ BitVec 32)) (loop1! (_ BitVec 32)) (sn! (_ BitVec 32)) (x (_ BitVec 32)) (n1 (_ BitVec 32)) (loop1 (_ BitVec 32)) (sn (_ BitVec 32))) Bool
       ( and ( = sn! ( bvadd sn #x00000002 ) ) ( = x! ( bvadd x #x00000001 ) ) ( = loop1! loop1 ) ( = n1! n1 ) ))
(define-fun post_fun ((x (_ BitVec 32)) (n1 (_ BitVec 32)) (loop1 (_ BitVec 32)) (sn (_ BitVec 32))) Bool
       ( or ( = sn ( bvmul #x00000002 x ) ) ( = sn #x00000000 ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

