(set-logic BV)

(synth-inv inv_fun ((z (_ BitVec 64))(x (_ BitVec 64))(y (_ BitVec 64))))

(define-fun pre_fun ((z (_ BitVec 64)) (x (_ BitVec 64)) (y (_ BitVec 64))) Bool
       ( and ( = x #x0000000000000000 ) ( = y #x00000000000000c8 ) ( = z #x0000000000000190 ) ))
(define-fun trans_fun ((z! (_ BitVec 64)) (y! (_ BitVec 64)) (x! (_ BitVec 64)) (z (_ BitVec 64)) (x (_ BitVec 64)) (y (_ BitVec 64))) Bool
       ( and ( = x! ( bvadd #x0000000000000001 x ) ) ( = y! ( ite ( bvslt x #x00000000000000c8 ) ( bvadd y #x0000000000000001 ) y ) ) ( = z! ( ite ( bvslt x #x00000000000000c8 ) z ( bvadd z #x0000000000000002 ) ) ) ))
(define-fun post_fun ((z (_ BitVec 64)) (x (_ BitVec 64)) (y (_ BitVec 64))) Bool
       ( let ( ( a!1 ( and ( bvsle #x0000000000000190 y ) ( not ( = z ( bvmul #x0000000000000002 x ) ) ) ) ) ) ( not a!1 ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

