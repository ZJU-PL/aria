(set-logic BV)

(synth-inv inv_fun ((y (_ BitVec 32))(x (_ BitVec 32))))

(define-fun pre_fun ((y (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( = x #x00000000 ))
(define-fun trans_fun ((y! (_ BitVec 32)) (x! (_ BitVec 32)) (y (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( let ( ( a!1 ( = x! ( ite ( bvult ( bvudiv x #x00000005 ) #x000000c8 ) ( bvadd x #x00000001 ) ( bvadd x #x00000005 ) ) ) ) ) ( and a!1 ( = y! ( ite ( = x #x000003e8 ) #x00000000 y ) ) ) ))
(define-fun post_fun ((y (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( not ( and ( bvule #x000007d0 x ) ( = y #x00000000 ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

