(set-logic BV)

(synth-inv inv_fun ((w (_ BitVec 64))(x (_ BitVec 64))(y (_ BitVec 64))(z (_ BitVec 64))))

(define-fun pre_fun ((w (_ BitVec 64)) (x (_ BitVec 64)) (y (_ BitVec 64)) (z (_ BitVec 64))) Bool
       ( and ( = x #x0000000000000000 ) ( = y #x0000000000000000 ) ( = z #x0000000000000000 ) ( = w #x0000000000000000 ) ))
(define-fun trans_fun ((w! (_ BitVec 64)) (z! (_ BitVec 64)) (y! (_ BitVec 64)) (x! (_ BitVec 64)) (w (_ BitVec 64)) (x (_ BitVec 64)) (y (_ BitVec 64)) (z (_ BitVec 64))) Bool
       ( let ( ( a!1 ( ite ( bvsgt ( bvsub y ( bvmul #x000000000000000a x ) ) #x0000000000000000 ) ( bvadd z #x0000000000000001 ) z ) ) ( a!2 ( ite ( bvsgt ( bvsub y ( bvmul #x000000000000000a x ) ) #x0000000000000000 ) w ( bvadd w #x0000000000000001 ) ) ) ) ( and ( = x! ( bvadd x #x0000000000000001 ) ) ( = y! ( bvadd y x ) ) ( = z! a!1 ) ( = w! a!2 ) ) ))
(define-fun post_fun ((w (_ BitVec 64)) (x (_ BitVec 64)) (y (_ BitVec 64)) (z (_ BitVec 64))) Bool
       ( not ( and ( not ( bvsle x #x0000000000000064 ) ) ( not ( bvsle z w ) ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

