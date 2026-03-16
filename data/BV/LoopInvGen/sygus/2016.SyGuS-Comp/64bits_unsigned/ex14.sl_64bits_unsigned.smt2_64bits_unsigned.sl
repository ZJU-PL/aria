(set-logic BV)

(synth-inv inv_fun ((y (_ BitVec 64))(x (_ BitVec 64))))

(define-fun pre_fun ((y (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( = x #x0000000000000001 ))
(define-fun trans_fun ((y! (_ BitVec 64)) (x! (_ BitVec 64)) (y (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( and ( bvule x #x000000000000000a ) ( = y! ( bvsub #x000000000000000a x ) ) ( = x! ( bvadd x #x0000000000000001 ) ) ))
(define-fun post_fun ((y (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( let ( ( a!1 ( and ( = ( ( _ extract (_ bv63 64) (_ bv4 64) ) x ) #x000000000000000 ) ( bvule ( ( _ extract (_ bv3 64) (_ bv0 64) ) x ) #xa ) ( = y ( bvadd #x000000000000000a ( bvmul #xffffffffffffffff x ) ) ) ( bvule #x000000000000000a y ) ) ) ) ( not a!1 ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

