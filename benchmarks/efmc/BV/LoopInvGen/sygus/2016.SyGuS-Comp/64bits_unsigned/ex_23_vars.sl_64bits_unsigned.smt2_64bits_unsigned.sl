(set-logic BV)

(synth-inv inv_fun ((x3 (_ BitVec 64))(x2 (_ BitVec 64))(x1 (_ BitVec 64))(c (_ BitVec 64))(z (_ BitVec 64))(y (_ BitVec 64))))

(define-fun pre_fun ((x3 (_ BitVec 64)) (x2 (_ BitVec 64)) (x1 (_ BitVec 64)) (c (_ BitVec 64)) (z (_ BitVec 64)) (y (_ BitVec 64))) Bool
       ( and ( = c #x0000000000000000 ) ( bvuge y #x0000000000000000 ) ( bvuge #x000000000000007f y ) ( = z ( bvmul #x0000000000000024 y ) ) ))
(define-fun trans_fun ((x3! (_ BitVec 64)) (x2! (_ BitVec 64)) (x1! (_ BitVec 64)) (c! (_ BitVec 64)) (z! (_ BitVec 64)) (y! (_ BitVec 64)) (x3 (_ BitVec 64)) (x2 (_ BitVec 64)) (x1 (_ BitVec 64)) (c (_ BitVec 64)) (z (_ BitVec 64)) (y (_ BitVec 64))) Bool
       ( and ( bvult c #x0000000000000024 ) ( = z! ( bvadd z #x0000000000000001 ) ) ( = c! ( bvadd c #x0000000000000001 ) ) ( = y! y ) ))
(define-fun post_fun ((x3 (_ BitVec 64)) (x2 (_ BitVec 64)) (x1 (_ BitVec 64)) (c (_ BitVec 64)) (z (_ BitVec 64)) (y (_ BitVec 64))) Bool
       ( not ( and ( not ( bvule #x0000000000000024 c ) ) ( bvule #x0000000000001200 z ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

