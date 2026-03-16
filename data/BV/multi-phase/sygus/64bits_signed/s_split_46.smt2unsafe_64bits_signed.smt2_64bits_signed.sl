(set-logic BV)

(synth-inv inv_fun ((y (_ BitVec 64))(x (_ BitVec 64))))

(define-fun pre_fun ((y (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( = x #x0000000000000000 ))
(define-fun trans_fun ((y! (_ BitVec 64)) (x! (_ BitVec 64)) (y (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( let ( ( a!1 ( = x! ( ite ( bvslt ( bvsdiv x #x0000000000000005 ) #x00000000000000c8 ) ( bvadd x #x0000000000000001 ) ( bvadd x #x0000000000000005 ) ) ) ) ) ( and a!1 ( = y! ( ite ( = x #x00000000000003e8 ) #x0000000000000000 y ) ) ) ))
(define-fun post_fun ((y (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( not ( and ( bvsle #x00000000000007d0 x ) ( = y #x0000000000000000 ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

