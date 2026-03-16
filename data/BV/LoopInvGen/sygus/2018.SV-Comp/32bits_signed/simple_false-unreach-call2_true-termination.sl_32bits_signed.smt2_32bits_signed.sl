(set-logic BV)

(synth-inv inv_fun ((x (_ BitVec 32))))

(define-fun pre_fun ((x (_ BitVec 32))) Bool
       ( bvsge x #x00000000 ))
(define-fun trans_fun ((x! (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( and ( bvslt x #x0fffffff ) ( = x! ( bvadd x #x00000001 ) ) ))
(define-fun post_fun ((x (_ BitVec 32))) Bool
       ( or ( not ( bvsle #x0fffffff x ) ) ( not ( bvsle x #x0fffffff ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

