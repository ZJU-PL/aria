(set-logic BV)

(synth-inv inv_fun ((t (_ BitVec 64))(k (_ BitVec 64))(j (_ BitVec 64))))

(define-fun pre_fun ((t (_ BitVec 64)) (k (_ BitVec 64)) (j (_ BitVec 64))) Bool
       ( and ( = j #x0000000000000002 ) ( = k #x0000000000000000 ) ))
(define-fun trans_fun ((t! (_ BitVec 64)) (k! (_ BitVec 64)) (j! (_ BitVec 64)) (t (_ BitVec 64)) (k (_ BitVec 64)) (j (_ BitVec 64))) Bool
       ( let ( ( a!1 ( or ( and ( = t #x0000000000000000 ) ( = j! ( bvadd j #x0000000000000004 ) ) ( = k! k ) ( = t! t ) ) ( and ( not ( = t #x0000000000000000 ) ) ( = j! ( bvadd j #x0000000000000002 ) ) ( = k! ( bvadd k #x0000000000000001 ) ) ( = t! t ) ) ) ) ) ( and a!1 ) ))
(define-fun post_fun ((t (_ BitVec 64)) (k (_ BitVec 64)) (j (_ BitVec 64))) Bool
       ( or ( = k #x0000000000000000 ) ( = j ( bvadd #x0000000000000002 ( bvmul #x0000000000000002 k ) ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

