(set-logic BV)

(synth-inv inv_fun ((x3 (_ BitVec 32))(x2 (_ BitVec 32))(x1 (_ BitVec 32))(c (_ BitVec 32))(z (_ BitVec 32))(y (_ BitVec 32))))

(define-fun pre_fun ((x3 (_ BitVec 32)) (x2 (_ BitVec 32)) (x1 (_ BitVec 32)) (c (_ BitVec 32)) (z (_ BitVec 32)) (y (_ BitVec 32))) Bool
       ( and ( = c #x00000000 ) ( bvuge y #x00000000 ) ( bvuge #x0000007f y ) ( = z ( bvmul #x00000024 y ) ) ))
(define-fun trans_fun ((x3! (_ BitVec 32)) (x2! (_ BitVec 32)) (x1! (_ BitVec 32)) (c! (_ BitVec 32)) (z! (_ BitVec 32)) (y! (_ BitVec 32)) (x3 (_ BitVec 32)) (x2 (_ BitVec 32)) (x1 (_ BitVec 32)) (c (_ BitVec 32)) (z (_ BitVec 32)) (y (_ BitVec 32))) Bool
       ( and ( bvult c #x00000024 ) ( = z! ( bvadd z #x00000001 ) ) ( = c! ( bvadd c #x00000001 ) ) ( = y! y ) ))
(define-fun post_fun ((x3 (_ BitVec 32)) (x2 (_ BitVec 32)) (x1 (_ BitVec 32)) (c (_ BitVec 32)) (z (_ BitVec 32)) (y (_ BitVec 32))) Bool
       ( not ( and ( not ( bvule #x00000024 c ) ) ( bvule #x00001200 z ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

