(set-logic BV)

(synth-inv inv_fun ((t (_ BitVec 32))(k (_ BitVec 32))(j (_ BitVec 32))))

(define-fun pre_fun ((t (_ BitVec 32)) (k (_ BitVec 32)) (j (_ BitVec 32))) Bool
       ( and ( = j #x00000002 ) ( = k #x00000000 ) ))
(define-fun trans_fun ((t! (_ BitVec 32)) (k! (_ BitVec 32)) (j! (_ BitVec 32)) (t (_ BitVec 32)) (k (_ BitVec 32)) (j (_ BitVec 32))) Bool
       ( let ( ( a!1 ( or ( and ( = t #x00000000 ) ( = j! ( bvadd j #x00000004 ) ) ( = k! k ) ( = t! t ) ) ( and ( not ( = t #x00000000 ) ) ( = j! ( bvadd j #x00000002 ) ) ( = k! ( bvadd k #x00000001 ) ) ( = t! t ) ) ) ) ) ( and a!1 ) ))
(define-fun post_fun ((t (_ BitVec 32)) (k (_ BitVec 32)) (j (_ BitVec 32))) Bool
       ( or ( = k #x00000000 ) ( = j ( bvadd #x00000002 ( bvmul #x00000002 k ) ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

