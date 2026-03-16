(set-logic BV)

(synth-inv inv_fun ((x (_ BitVec 64))))

(define-fun pre_fun ((x (_ BitVec 64))) Bool
       ( = x x ))
(define-fun trans_fun ((x (_ BitVec 64)) (x! (_ BitVec 64))) Bool
       ( and ( bvslt x #x0fffffff ) ( = x! ( bvadd x #x00000001 ) ) ))
(define-fun post_fun ((x (_ BitVec 64))) Bool
       ( or ( bvsge x #x0fffffff ) ( bvslt x #x0fffffff ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

