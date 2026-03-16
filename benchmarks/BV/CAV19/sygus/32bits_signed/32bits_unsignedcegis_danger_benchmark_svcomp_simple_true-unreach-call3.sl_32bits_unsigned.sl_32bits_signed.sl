(set-logic BV)

(synth-inv inv_fun ((x (_ BitVec 32))(N (_ BitVec 32))))

(define-fun pre_fun ((x (_ BitVec 32)) (N (_ BitVec 32))) Bool
       ( and ( = x #x00000000 ) ( = N ( bvurem N #x0000ffff ) ) ))
(define-fun trans_fun ((x (_ BitVec 32)) (N (_ BitVec 32)) (x! (_ BitVec 32)) (N! (_ BitVec 32))) Bool
       ( and ( bvslt x N ) ( = x! ( bvadd x #x00000002 ) ) ( = N! N ) ))
(define-fun post_fun ((x (_ BitVec 32)) (N (_ BitVec 32))) Bool
       ( or ( = #x00000000 ( bvurem x #x00000002 ) ) ( bvslt x N ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

