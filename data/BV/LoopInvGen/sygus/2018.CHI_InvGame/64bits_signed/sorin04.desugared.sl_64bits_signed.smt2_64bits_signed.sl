(set-logic BV)

(synth-inv inv_fun ((c (_ BitVec 64))(a (_ BitVec 64))(i (_ BitVec 64))))

(define-fun pre_fun ((c (_ BitVec 64)) (a (_ BitVec 64)) (i (_ BitVec 64))) Bool
       ( and ( = i #x0000000000000000 ) ( = a #x0000000000000000 ) ( = c #x0000000000000000 ) ))
(define-fun trans_fun ((c! (_ BitVec 64)) (a! (_ BitVec 64)) (i! (_ BitVec 64)) (c (_ BitVec 64)) (a (_ BitVec 64)) (i (_ BitVec 64))) Bool
       ( and ( bvslt i #x000000000000000a ) ( = c! ( bvadd c #x0000000000000001 ( bvmul #x0000000000000006 a ) ) ) ( = a! ( bvadd a ( bvadd i #x0000000000000001 ) ) ) ( = i! ( bvadd i #x0000000000000001 ) ) ))
(define-fun post_fun ((c (_ BitVec 64)) (a (_ BitVec 64)) (i (_ BitVec 64))) Bool
       ( or ( not ( bvsle #x000000000000000a i ) ) ( = c ( bvmul i i i ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

