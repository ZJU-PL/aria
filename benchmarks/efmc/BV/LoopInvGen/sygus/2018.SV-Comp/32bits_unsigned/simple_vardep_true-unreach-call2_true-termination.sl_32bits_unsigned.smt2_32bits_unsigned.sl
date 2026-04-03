(set-logic BV)

(synth-inv inv_fun ((k (_ BitVec 32))(j (_ BitVec 32))(i (_ BitVec 32))))

(define-fun pre_fun ((k (_ BitVec 32)) (j (_ BitVec 32)) (i (_ BitVec 32))) Bool
       ( and ( = i #x00000000 ) ( = j #x00000000 ) ( = k #x00000000 ) ))
(define-fun trans_fun ((k! (_ BitVec 32)) (j! (_ BitVec 32)) (i! (_ BitVec 32)) (k (_ BitVec 32)) (j (_ BitVec 32)) (i (_ BitVec 32))) Bool
       ( and ( bvult k #x0fffffff ) ( = i! ( bvadd i #x00000001 ) ) ( = j! ( bvadd j #x00000002 ) ) ( = k! ( bvadd k #x00000003 ) ) ))
(define-fun post_fun ((k (_ BitVec 32)) (j (_ BitVec 32)) (i (_ BitVec 32))) Bool
       ( or ( not ( bvule #x0fffffff k ) ) ( and ( = k ( bvmul #x00000003 i ) ) ( = j ( bvmul #x00000002 i ) ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

