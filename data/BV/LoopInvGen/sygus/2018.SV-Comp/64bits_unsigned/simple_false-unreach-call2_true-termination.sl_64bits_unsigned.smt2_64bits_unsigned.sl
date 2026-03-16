(set-logic BV)

(synth-inv inv_fun ((x (_ BitVec 64))))

(define-fun pre_fun ((x (_ BitVec 64))) Bool
       ( bvuge x #x0000000000000000 ))
(define-fun trans_fun ((x! (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( and ( bvult x #x000000000fffffff ) ( = x! ( bvadd x #x0000000000000001 ) ) ))
(define-fun post_fun ((x (_ BitVec 64))) Bool
       ( or ( not ( bvule #x000000000fffffff x ) ) ( not ( = ( ( _ extract (_ bv63 64) (_ bv28 64) ) x ) #x000000000 ) ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

