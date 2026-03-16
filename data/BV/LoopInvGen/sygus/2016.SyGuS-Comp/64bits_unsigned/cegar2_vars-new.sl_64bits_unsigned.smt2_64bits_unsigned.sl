(set-logic BV)

(synth-inv inv_fun ((z3 (_ BitVec 64))(z2 (_ BitVec 64))(z1 (_ BitVec 64))(m (_ BitVec 64))(n (_ BitVec 64))(x (_ BitVec 64))))

(define-fun pre_fun ((z3 (_ BitVec 64)) (z2 (_ BitVec 64)) (z1 (_ BitVec 64)) (m (_ BitVec 64)) (n (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( and ( = x #x0000000000000001 ) ( = m #x0000000000000001 ) ))
(define-fun trans_fun ((z3! (_ BitVec 64)) (z2! (_ BitVec 64)) (z1! (_ BitVec 64)) (m! (_ BitVec 64)) (n! (_ BitVec 64)) (x! (_ BitVec 64)) (z3 (_ BitVec 64)) (z2 (_ BitVec 64)) (z1 (_ BitVec 64)) (m (_ BitVec 64)) (n (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( let ( ( a!1 ( and ( bvult x n ) ( = x! ( bvadd x #x0000000000000001 ) ) ( = n! n ) ) ) ) ( and ( or ( and a!1 ( = m! m ) ) ( and a!1 ( = m! x ) ) ) ) ))
(define-fun post_fun ((z3 (_ BitVec 64)) (z2 (_ BitVec 64)) (z1 (_ BitVec 64)) (m (_ BitVec 64)) (n (_ BitVec 64)) (x (_ BitVec 64))) Bool
       ( let ( ( a!1 ( and ( bvule n x ) ( not ( = ( ( _ extract (_ bv63 64) (_ bv1 64) ) n ) #b000000000000000000000000000000000000000000000000000000000000000 ) ) ( or ( bvule n m ) ( not ( bvule #x0000000000000001 m ) ) ) ) ) ) ( not a!1 ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

