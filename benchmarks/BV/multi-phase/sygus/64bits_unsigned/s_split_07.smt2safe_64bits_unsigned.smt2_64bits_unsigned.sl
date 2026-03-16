(set-logic BV)

(synth-inv inv_fun ((v (_ BitVec 64))(y (_ BitVec 64))(x (_ BitVec 64))(z (_ BitVec 64))))

(define-fun pre_fun ((v (_ BitVec 64)) (y (_ BitVec 64)) (x (_ BitVec 64)) (z (_ BitVec 64))) Bool
       ( and ( bvugt x y ) ( bvugt y z ) ( = v #x0000000000000000 ) ))
(define-fun trans_fun ((v! (_ BitVec 64)) (z! (_ BitVec 64)) (y! (_ BitVec 64)) (x! (_ BitVec 64)) (v (_ BitVec 64)) (y (_ BitVec 64)) (x (_ BitVec 64)) (z (_ BitVec 64))) Bool
       ( and ( = x! ( bvadd x #x0000000000000001 ) ) ( = y! ( bvadd y #x0000000000000003 ) ) ( = z! ( bvadd z #x0000000000000002 ) ) ( = v! ( ite ( bvult x y ) ( bvadd v #x0000000000000001 ) v ) ) ))
(define-fun post_fun ((v (_ BitVec 64)) (y (_ BitVec 64)) (x (_ BitVec 64)) (z (_ BitVec 64))) Bool
       ( let ( ( a!1 ( = ( ( _ extract (_ bv63 64) (_ bv17 64) ) ( bvadd z ( bvmul #xffffffffffffffff x ) ) ) #b00000000000000000000000000000000000000000000000 ) ) ( a!2 ( bvule ( bvadd ( ( _ extract (_ bv16 64) (_ bv0 64) ) z ) ( bvmul #b11111111111111111 ( ( _ extract (_ bv16 64) (_ bv0 64) ) x ) ) ) #b10001101101010011 ) ) ) ( not ( and ( not ( and a!1 a!2 ) ) ( = v #x0000000000000000 ) ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

