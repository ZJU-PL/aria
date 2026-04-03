(set-logic BV)

(synth-inv inv_fun ((x3 (_ BitVec 32))(x2 (_ BitVec 32))(x1 (_ BitVec 32))(c (_ BitVec 32))(z (_ BitVec 32))(y (_ BitVec 32))))

(define-fun pre_fun ((x3 (_ BitVec 32)) (x2 (_ BitVec 32)) (x1 (_ BitVec 32)) (c (_ BitVec 32)) (z (_ BitVec 32)) (y (_ BitVec 32))) Bool
       ( and ( = c #x00000000 ) ( bvsge y #x00000000 ) ( bvsge #x0000007f y ) ( = z ( bvmul #x00000024 y ) ) ))
(define-fun trans_fun ((x3! (_ BitVec 32)) (x2! (_ BitVec 32)) (x1! (_ BitVec 32)) (c! (_ BitVec 32)) (z! (_ BitVec 32)) (y! (_ BitVec 32)) (x3 (_ BitVec 32)) (x2 (_ BitVec 32)) (x1 (_ BitVec 32)) (c (_ BitVec 32)) (z (_ BitVec 32)) (y (_ BitVec 32))) Bool
       ( and ( bvslt c #x00000024 ) ( = z! ( bvadd z #x00000001 ) ) ( = c! ( bvadd c #x00000001 ) ) ( = y! y ) ))
(define-fun post_fun ((x3 (_ BitVec 32)) (x2 (_ BitVec 32)) (x1 (_ BitVec 32)) (c (_ BitVec 32)) (z (_ BitVec 32)) (y (_ BitVec 32))) Bool
       ( let ( ( a!1 ( and ( not ( bvsle #x00000024 c ) ) ( or ( not ( bvsle #x00000000 z ) ) ( bvsle #x00001200 z ) ) ) ) ) ( not a!1 ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

