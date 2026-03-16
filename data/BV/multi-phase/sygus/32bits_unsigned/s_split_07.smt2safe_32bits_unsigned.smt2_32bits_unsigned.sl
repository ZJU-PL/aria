(set-logic BV)

(synth-inv inv_fun ((v (_ BitVec 32))(y (_ BitVec 32))(x (_ BitVec 32))(z (_ BitVec 32))))

(define-fun pre_fun ((v (_ BitVec 32)) (y (_ BitVec 32)) (x (_ BitVec 32)) (z (_ BitVec 32))) Bool
       ( and ( bvugt x y ) ( bvugt y z ) ( = v #x00000000 ) ))
(define-fun trans_fun ((v! (_ BitVec 32)) (z! (_ BitVec 32)) (y! (_ BitVec 32)) (x! (_ BitVec 32)) (v (_ BitVec 32)) (y (_ BitVec 32)) (x (_ BitVec 32)) (z (_ BitVec 32))) Bool
       ( and ( = x! ( bvadd x #x00000001 ) ) ( = y! ( bvadd y #x00000003 ) ) ( = z! ( bvadd z #x00000002 ) ) ( = v! ( ite ( bvult x y ) ( bvadd v #x00000001 ) v ) ) ))
(define-fun post_fun ((v (_ BitVec 32)) (y (_ BitVec 32)) (x (_ BitVec 32)) (z (_ BitVec 32))) Bool
       ( let ( ( a!1 ( = ( ( _ extract (_ bv31 32) (_ bv17 32) ) ( bvadd z ( bvmul #xffffffff x ) ) ) #b000000000000000 ) ) ( a!2 ( bvule ( bvadd ( ( _ extract (_ bv16 32) (_ bv0 32) ) z ) ( bvmul #b11111111111111111 ( ( _ extract (_ bv16 32) (_ bv0 32) ) x ) ) ) #b10001101101010011 ) ) ) ( not ( and ( not ( and a!1 a!2 ) ) ( = v #x00000000 ) ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

