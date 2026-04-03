(set-logic BV)

(synth-inv inv_fun ((y (_ BitVec 32))(x (_ BitVec 32))))

(define-fun pre_fun ((y (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( and ( = x #x0000c350 ) ( = y #x00000000 ) ))
(define-fun trans_fun ((y! (_ BitVec 32)) (x! (_ BitVec 32)) (y (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( and ( = x! ( ite ( bvuge y x ) ( bvadd x #x00000005 ) x ) ) ( = y! ( ite ( bvuge y x ) y ( bvadd y #x00000001 ) ) ) ))
(define-fun post_fun ((y (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( let ( ( a!1 ( not ( and ( = ( ( _ extract (_ bv31 32) (_ bv16 32) ) y ) #x0000 ) ( bvule ( ( _ extract (_ bv15 32) (_ bv0 32) ) y ) #xc350 ) ) ) ) ( a!2 ( = ( ( _ extract (_ bv31 32) (_ bv3 32) ) ( bvadd x ( bvmul #xffffffff y ) ) ) #b00000000000000000000000000000 ) ) ( a!3 ( bvule ( bvadd ( ( _ extract (_ bv2 32) (_ bv0 32) ) x ) ( bvmul #b111 ( ( _ extract (_ bv2 32) (_ bv0 32) ) y ) ) ) #b101 ) ) ) ( not ( and a!1 ( not ( and a!2 a!3 ) ) ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

