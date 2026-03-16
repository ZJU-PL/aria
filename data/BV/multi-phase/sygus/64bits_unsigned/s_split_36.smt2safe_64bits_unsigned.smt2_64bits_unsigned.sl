(set-logic BV)

(synth-inv inv_fun ((y (_ BitVec 64))(x (_ BitVec 64))))

(define-fun pre_fun ((y (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( and ( = x ( bvadd #x0000000000000001 ( bvnot #x0000000000002710 ) ) ) ( = y #x0000000000000000 ) ))
(define-fun trans_fun ((y! (_ BitVec 64)) (x! (_ BitVec 64)) (y (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( let ( ( a!1 ( = y! ( ite ( bvuge y x ) ( bvadd #x0000000000000001 ( bvnot x ) ) ( bvadd y #x0000000000000002 ) ) ) ) ) ( and ( = x! ( ite ( bvuge y x ) ( bvadd x #x0000000000000001 ) x ) ) a!1 ) ))
(define-fun post_fun ((y (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( bvule ( bvadd #xffffffffffffffff y ) x ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

