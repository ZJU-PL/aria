(set-logic BV)

(synth-inv inv_fun ((z (_ BitVec 32))(x (_ BitVec 32))(y (_ BitVec 32))))

(define-fun pre_fun ((z (_ BitVec 32)) (x (_ BitVec 32)) (y (_ BitVec 32))) Bool
       ( and ( = x #x00000001 ) ( = y #x00000000 ) ( = z #x00000000 ) ))
(define-fun trans_fun ((z! (_ BitVec 32)) (y! (_ BitVec 32)) (x! (_ BitVec 32)) (z (_ BitVec 32)) (x (_ BitVec 32)) (y (_ BitVec 32))) Bool
       ( and ( = x! ( bvadd #x00000001 ( bvnot x ) ) ) ( = y! ( ite ( bvsgt x #x00000000 ) ( bvadd y #x00000001 ) y ) ) ( = z! ( ite ( bvsgt x #x00000000 ) z ( bvadd z #x00000001 ) ) ) ))
(define-fun post_fun ((z (_ BitVec 32)) (x (_ BitVec 32)) (y (_ BitVec 32))) Bool
       ( not ( and ( = x #x00000001 ) ( = y #x1467b6dd ) ( not ( = z #x1467b6dd ) ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

