(set-logic BV)

(synth-inv inv_fun ((y (_ BitVec 32))(x (_ BitVec 32))))

(define-fun pre_fun ((y (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( and ( = x ( bvadd #x00000001 ( bvnot #x00002710 ) ) ) ( = y #x00000000 ) ))
(define-fun trans_fun ((y! (_ BitVec 32)) (x! (_ BitVec 32)) (y (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( let ( ( a!1 ( = y! ( ite ( bvuge y x ) ( bvadd #x00000001 ( bvnot x ) ) ( bvadd y #x00000002 ) ) ) ) ) ( and ( = x! ( ite ( bvuge y x ) ( bvadd x #x00000001 ) x ) ) a!1 ) ))
(define-fun post_fun ((y (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( not ( bvule ( bvadd #xffffffff y ) x ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

