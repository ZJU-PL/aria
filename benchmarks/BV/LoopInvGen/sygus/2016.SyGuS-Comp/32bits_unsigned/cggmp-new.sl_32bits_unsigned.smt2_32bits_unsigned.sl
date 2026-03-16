(set-logic BV)

(synth-inv inv_fun ((j (_ BitVec 32))(i (_ BitVec 32))))

(define-fun pre_fun ((j (_ BitVec 32)) (i (_ BitVec 32))) Bool
       ( and ( = i #x00000001 ) ( = j #x00000014 ) ))
(define-fun trans_fun ((j! (_ BitVec 32)) (i! (_ BitVec 32)) (j (_ BitVec 32)) (i (_ BitVec 32))) Bool
       ( and ( bvuge j i ) ( = i! ( bvadd i #x00000002 ) ) ( = j! ( bvsub j #x00000001 ) ) ))
(define-fun post_fun ((j (_ BitVec 32)) (i (_ BitVec 32))) Bool
       ( not ( and ( not ( bvule i j ) ) ( not ( = j #x0000000d ) ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

