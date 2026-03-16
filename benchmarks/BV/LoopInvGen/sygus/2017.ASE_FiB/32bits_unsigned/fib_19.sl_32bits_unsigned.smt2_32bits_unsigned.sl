(set-logic BV)

(synth-inv inv_fun ((y (_ BitVec 32))(x (_ BitVec 32))(m (_ BitVec 32))(n (_ BitVec 32))))

(define-fun pre_fun ((y (_ BitVec 32)) (x (_ BitVec 32)) (m (_ BitVec 32)) (n (_ BitVec 32))) Bool
       ( and ( bvuge n #x00000000 ) ( bvuge m #x00000000 ) ( bvult m n ) ( = x #x00000000 ) ( = y m ) ))
(define-fun trans_fun ((y! (_ BitVec 32)) (x! (_ BitVec 32)) (m! (_ BitVec 32)) (n! (_ BitVec 32)) (y (_ BitVec 32)) (x (_ BitVec 32)) (m (_ BitVec 32)) (n (_ BitVec 32))) Bool
       ( let ( ( a!1 ( or ( and ( bvult x n ) ( bvule ( bvadd x #x00000001 ) m ) ( = x! ( bvadd x #x00000001 ) ) ( = y! y ) ( = n! n ) ( = m! m ) ) ( and ( bvult x n ) ( bvugt ( bvadd x #x00000001 ) m ) ( = x! ( bvadd x #x00000001 ) ) ( = y! ( bvadd y #x00000001 ) ) ( = n! n ) ( = m! m ) ) ( and ( bvuge x n ) ( = x! x ) ( = y! y ) ( = n! n ) ( = m! m ) ) ) ) ) ( and a!1 ) ))
(define-fun post_fun ((y (_ BitVec 32)) (x (_ BitVec 32)) (m (_ BitVec 32)) (n (_ BitVec 32))) Bool
       ( or ( not ( bvule n x ) ) ( = y n ) ))

(inv-constraint inv_fun pre_fun trans_fun post_fun)

(check-synth)

