(set-logic BV)

(synth-inv inv_fun ((x (_ BitVec 64))(N (_ BitVec 64))))

(define-fun pre_fun ((x (_ BitVec 64)) (N (_ BitVec 64))) Bool
       ( and ( = x #x00000000 ) ( = N ( bvurem N #x0000ffff ) ) ))
(define-fun trans_fun ((x (_ BitVec 64)) (N (_ BitVec 64)) (x! (_ BitVec 64)) (N! (_ BitVec 64))) Bool
       ( and ( bvult x N ) ( = x! ( bvadd x #x00000002 ) ) ( = N! N ) ))
(define-fun post_fun ((x (_ BitVec 64)) (N (_ BitVec 64))) Bool
       ( or ( = #x00000000 ( bvurem x #x00000002 ) ) ( bvult x N ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

