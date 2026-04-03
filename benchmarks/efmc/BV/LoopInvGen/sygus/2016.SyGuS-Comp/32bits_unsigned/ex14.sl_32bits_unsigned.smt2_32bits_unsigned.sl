(set-logic BV)

(synth-inv inv_fun ((y (_ BitVec 32))(x (_ BitVec 32))))

(define-fun pre_fun ((y (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( = x #x00000001 ))
(define-fun trans_fun ((y! (_ BitVec 32)) (x! (_ BitVec 32)) (y (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( and ( bvule x #x0000000a ) ( = y! ( bvsub #x0000000a x ) ) ( = x! ( bvadd x #x00000001 ) ) ))
(define-fun post_fun ((y (_ BitVec 32)) (x (_ BitVec 32))) Bool
       ( let ( ( a!1 ( and ( = ( ( _ extract (_ bv31 32) (_ bv4 32) ) x ) #x0000000 ) ( bvule ( ( _ extract (_ bv3 32) (_ bv0 32) ) x ) #xa ) ( = y ( bvadd #x0000000a ( bvmul #xffffffff x ) ) ) ( bvule #x0000000a y ) ) ) ) ( not a!1 ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

