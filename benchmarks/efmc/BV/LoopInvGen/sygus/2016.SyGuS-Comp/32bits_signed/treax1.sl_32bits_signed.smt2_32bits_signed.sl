(set-logic BV)

(synth-inv inv_fun ((y (_ BitVec 32))(x (_ BitVec 32))))

(define-fun pre_fun ((y (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( = x #x00000001 ))
(define-fun trans_fun ((y! (_ BitVec 32)) (x! (_ BitVec 32)) (y (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( and ( bvslt x y ) ( = x! ( bvadd x x ) ) ))
(define-fun post_fun ((y (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( or ( not ( bvsle y x ) ) ( bvsle #x00000001 x ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

