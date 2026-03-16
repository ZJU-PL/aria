(set-logic BV)

(synth-inv inv_fun ((y (_ BitVec 64))(x (_ BitVec 64))(m (_ BitVec 64))(n (_ BitVec 64))))

(define-fun pre_fun ((y (_ BitVec 64)) (x (_ BitVec 64)) (m (_ BitVec 64)) (n (_ BitVec 64))) Bool
       ( and ( bvuge n #x0000000000000000 ) ( bvuge m #x0000000000000000 ) ( bvult m n ) ( = x #x0000000000000000 ) ( = y m ) ))
(define-fun trans_fun ((y! (_ BitVec 64)) (x! (_ BitVec 64)) (m! (_ BitVec 64)) (n! (_ BitVec 64)) (y (_ BitVec 64)) (x (_ BitVec 64)) (m (_ BitVec 64)) (n (_ BitVec 64))) Bool
       ( let ( ( a!1 ( or ( and ( bvult x n ) ( bvule ( bvadd x #x0000000000000001 ) m ) ( = x! ( bvadd x #x0000000000000001 ) ) ( = y! y ) ( = n! n ) ( = m! m ) ) ( and ( bvult x n ) ( bvugt ( bvadd x #x0000000000000001 ) m ) ( = x! ( bvadd x #x0000000000000001 ) ) ( = y! ( bvadd y #x0000000000000001 ) ) ( = n! n ) ( = m! m ) ) ( and ( bvuge x n ) ( = x! x ) ( = y! y ) ( = n! n ) ( = m! m ) ) ) ) ) ( and a!1 ) ))
(define-fun post_fun ((y (_ BitVec 64)) (x (_ BitVec 64)) (m (_ BitVec 64)) (n (_ BitVec 64))) Bool
       ( or ( not ( bvule n x ) ) ( = y n ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

