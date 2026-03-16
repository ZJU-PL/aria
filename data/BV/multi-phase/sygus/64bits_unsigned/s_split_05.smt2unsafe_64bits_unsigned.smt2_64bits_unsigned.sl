(set-logic BV)

(synth-inv inv_fun ((z (_ BitVec 64))(y (_ BitVec 64))(x (_ BitVec 64))))

(define-fun pre_fun ((z (_ BitVec 64)) (y (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( and ( bvugt x #x0000000000000000 ) ( bvult y #x0000000000000000 ) ( = z #x0000000000000001 ) ))
(define-fun trans_fun ((z! (_ BitVec 64)) (y! (_ BitVec 64)) (x! (_ BitVec 64)) (z (_ BitVec 64)) (y (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( and ( = x! ( bvadd x #x0000000000000001 ) ) ( = y! ( bvadd y #x0000000000000002 ) ) ( = z! ( ite ( bvuge y #x0000000000000000 ) ( bvmul z #x0000000000000002 ) z ) ) ))
(define-fun post_fun ((z (_ BitVec 64)) (y (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( let ( ( a!1 ( and ( bvule x y ) ( not ( = ( ( _ extract (_ bv63 64) (_ bv1 64) ) z ) #b000000000000000000000000000000000000000000000000000000000000000 ) ) ) ) ) ( not a!1 ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

