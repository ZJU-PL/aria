(set-logic BV)

(synth-inv inv_fun ((y (_ BitVec 64))(x (_ BitVec 64))))

(define-fun pre_fun ((y (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( and ( bvsge x #x0000000000000000 ) ( bvsle x #x000000000000000a ) ( bvsle y #x000000000000000a ) ( bvsge y #x0000000000000000 ) ))
(define-fun trans_fun ((y! (_ BitVec 64)) (x! (_ BitVec 64)) (y (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( and ( = x! ( bvadd x #x000000000000000a ) ) ( = y! ( bvadd y #x000000000000000a ) ) ))
(define-fun post_fun ((y (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( not ( and ( = x #x0000000000000014 ) ( = y #x0000000000000000 ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

