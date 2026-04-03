(set-logic BV)

(synth-inv inv_fun ((y (_ BitVec 64))(x (_ BitVec 64))))

(define-fun pre_fun ((y (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( and ( = x #x000000000000c350 ) ( = y #x0000000000000000 ) ))
(define-fun trans_fun ((y! (_ BitVec 64)) (x! (_ BitVec 64)) (y (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( and ( = x! ( ite ( bvsge y x ) ( bvadd x #x0000000000000005 ) x ) ) ( = y! ( ite ( bvsge y x ) y ( bvadd y #x0000000000000001 ) ) ) ))
(define-fun post_fun ((y (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( let ( ( a!1 ( not ( bvsle ( bvadd x ( bvmul #xffffffffffffffff y ) ) #x0000000000000005 ) ) ) ) ( not ( and ( not ( bvsle y #x000000000000c350 ) ) a!1 ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

