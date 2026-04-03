(set-logic BV)

(synth-inv inv_fun ((y (_ BitVec 32))(x (_ BitVec 32))))

(define-fun pre_fun ((y (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( = x #x00000001 ))
(define-fun trans_fun ((y! (_ BitVec 32)) (x! (_ BitVec 32)) (y (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( and ( bvule x #x00000064 ) ( = y! ( bvsub #x00000064 x ) ) ( = x! ( bvadd x #x00000001 ) ) ))
(define-fun post_fun ((y (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( let ( ( a!1 ( and ( = ( ( _ extract (_ bv31 32) (_ bv7 32) ) x ) #b0000000000000000000000000 ) ( bvule ( ( _ extract (_ bv6 32) (_ bv0 32) ) x ) #b1100100 ) ( = y ( bvadd #x00000064 ( bvmul #xffffffff x ) ) ) ( bvule #x00000064 y ) ) ) ) ( not a!1 ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

