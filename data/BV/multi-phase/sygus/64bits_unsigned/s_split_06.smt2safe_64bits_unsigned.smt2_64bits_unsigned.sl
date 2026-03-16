(set-logic BV)

(synth-inv inv_fun ((z (_ BitVec 64))(x (_ BitVec 64))(y (_ BitVec 64))))

(define-fun pre_fun ((z (_ BitVec 64)) (x (_ BitVec 64)) (y (_ BitVec 64))) Bool
       ( and ( = x #x0000000000000001 ) ( = y #x0000000000000000 ) ( = z #x0000000000000000 ) ))
(define-fun trans_fun ((z! (_ BitVec 64)) (y! (_ BitVec 64)) (x! (_ BitVec 64)) (z (_ BitVec 64)) (x (_ BitVec 64)) (y (_ BitVec 64))) Bool
       ( and ( = x! ( bvadd #x0000000000000001 ( bvnot x ) ) ) ( = y! ( ite ( bvugt x #x0000000000000000 ) ( bvadd y #x0000000000000001 ) y ) ) ( = z! ( ite ( bvugt x #x0000000000000000 ) z ( bvadd z #x0000000000000001 ) ) ) ))
(define-fun post_fun ((z (_ BitVec 64)) (x (_ BitVec 64)) (y (_ BitVec 64))) Bool
       ( not ( and ( = x #x0000000000000001 ) ( = y #x000000001467b6dd ) ( not ( = z #x000000001467b6dd ) ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

