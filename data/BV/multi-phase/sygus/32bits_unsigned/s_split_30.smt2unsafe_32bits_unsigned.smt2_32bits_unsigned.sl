(set-logic BV)

(synth-inv inv_fun ((w (_ BitVec 32))(x (_ BitVec 32))(z (_ BitVec 32))(y (_ BitVec 32))))

(define-fun pre_fun ((w (_ BitVec 32)) (x (_ BitVec 32)) (z (_ BitVec 32)) (y (_ BitVec 32))) Bool
       ( and ( = x #x00000034 ) ( = y #x00000061 ) ( = z ( bvadd #x00000001 ( bvnot #x0000004c ) ) ) ( = w #x00000000 ) ))
(define-fun trans_fun ((w! (_ BitVec 32)) (z! (_ BitVec 32)) (y! (_ BitVec 32)) (x! (_ BitVec 32)) (w (_ BitVec 32)) (x (_ BitVec 32)) (z (_ BitVec 32)) (y (_ BitVec 32))) Bool
       ( let ( ( a!1 ( bvadd ( bvmul ( bvadd #x00000001 ( bvnot #x00000005 ) ) x ) ( bvmul #x00000003 y ) ( bvmul #x00000004 z ) #x00000001 ( bvnot #x00002232 ) ) ) ) ( and ( = x! ( bvsub #x0000000d ( bvmul #x00000007 x ) ) ) ( = y! ( bvsub #x00000036 ( bvmul #x00000002 y ) ) ) ( = z! a!1 ) ( = w! ( ite ( bvugt z! #x00000000 ) ( bvsub w x ) w ) ) ) ))
(define-fun post_fun ((w (_ BitVec 32)) (x (_ BitVec 32)) (z (_ BitVec 32)) (y (_ BitVec 32))) Bool
       ( not ( and ( bvule #x00013c12 y ) ( not ( = w #x00000000 ) ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

