(set-logic BV)

(synth-inv inv_fun ((x (_ BitVec 32))(y (_ BitVec 32))))

(define-fun pre_fun ((x (_ BitVec 32)) (y (_ BitVec 32))) Bool
       ( = x #x00000000 ))
(define-fun trans_fun ((x (_ BitVec 32)) (y (_ BitVec 32)) (x! (_ BitVec 32)) (y! (_ BitVec 32))) Bool
       ( or ( and ( = ( bvurem y #x00000002 ) #x00000000 ) ( = x! ( bvadd x #x00000002 ) ) ( and ( = y! y ) ( bvult x #x00000063 ) ) ) ( and ( = x! ( bvadd #x00000001 x ) ) ( and ( = y! y ) ( bvult x #x00000063 ) ) ) ))
(define-fun post_fun ((x (_ BitVec 32)) (y (_ BitVec 32))) Bool
       ( or ( = ( bvurem x #x00000002 ) ( bvurem y #x00000002 ) ) ( bvult x #x00000063 ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

