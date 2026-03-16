(set-logic BV)

(synth-inv inv_fun ((y (_ BitVec 64))(x (_ BitVec 64))))

(define-fun pre_fun ((y (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( and ( = x #x000000000000c350 ) ( = y #x0000000000000000 ) ))
(define-fun trans_fun ((y! (_ BitVec 64)) (x! (_ BitVec 64)) (y (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( and ( = x! ( ite ( bvuge y x ) ( bvadd x #x0000000000000005 ) x ) ) ( = y! ( ite ( bvuge y x ) y ( bvadd y #x0000000000000001 ) ) ) ))
(define-fun post_fun ((y (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( let ( ( a!1 ( not ( and ( = ( ( _ extract (_ bv63 64) (_ bv16 64) ) y ) #x000000000000 ) ( bvule ( ( _ extract (_ bv15 64) (_ bv0 64) ) y ) #xc350 ) ) ) ) ( a!2 ( = ( ( _ extract (_ bv63 64) (_ bv3 64) ) ( bvadd x ( bvmul #xffffffffffffffff y ) ) ) #b0000000000000000000000000000000000000000000000000000000000000 ) ) ( a!3 ( bvule ( bvadd ( ( _ extract (_ bv2 64) (_ bv0 64) ) x ) ( bvmul #b111 ( ( _ extract (_ bv2 64) (_ bv0 64) ) y ) ) ) #b101 ) ) ) ( not ( and a!1 a!2 a!3 ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

