(set-logic BV)

(synth-inv inv_fun ((w (_ BitVec 64))(x (_ BitVec 64))(z (_ BitVec 64))(y (_ BitVec 64))))

(define-fun pre_fun ((w (_ BitVec 64)) (x (_ BitVec 64)) (z (_ BitVec 64)) (y (_ BitVec 64))) Bool
       ( and ( = x #x0000000000000034 ) ( = y #x0000000000000061 ) ( = z ( bvadd #x0000000000000001 ( bvnot #x000000000000004c ) ) ) ( = w #x0000000000000000 ) ))
(define-fun trans_fun ((w! (_ BitVec 64)) (z! (_ BitVec 64)) (y! (_ BitVec 64)) (x! (_ BitVec 64)) (w (_ BitVec 64)) (x (_ BitVec 64)) (z (_ BitVec 64)) (y (_ BitVec 64))) Bool
       ( let ( ( a!1 ( bvadd ( bvmul ( bvadd #x0000000000000001 ( bvnot #x0000000000000005 ) ) x ) ( bvmul #x0000000000000003 y ) ( bvmul #x0000000000000004 z ) #x0000000000000001 ( bvnot #x0000000000002232 ) ) ) ) ( and ( = x! ( bvsub #x000000000000000d ( bvmul #x0000000000000007 x ) ) ) ( = y! ( bvsub #x0000000000000036 ( bvmul #x0000000000000002 y ) ) ) ( = z! a!1 ) ( = w! ( ite ( bvugt z! #x0000000000000000 ) ( bvsub w x ) w ) ) ) ))
(define-fun post_fun ((w (_ BitVec 64)) (x (_ BitVec 64)) (z (_ BitVec 64)) (y (_ BitVec 64))) Bool
       ( not ( and ( bvule #x0000000000013c12 y ) ( = w #x0000000000000000 ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

